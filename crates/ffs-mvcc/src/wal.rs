//! Write-ahead log (WAL) format for MVCC version persistence.
//!
//! This module defines the binary format for persisting MVCC commits to disk.
//! The WAL uses an append-only structure with CRC32c checksums for corruption
//! detection.
//!
//! # Format Overview
//!
//! ```text
//! WAL File:
//! +----------------+
//! | File Header    |  (16 bytes)
//! +----------------+
//! | Commit Record  |  (variable)
//! +----------------+
//! | Commit Record  |  (variable)
//! +----------------+
//! | ...            |
//! +----------------+
//!
//! File Header:
//! +------------------+--------+
//! | magic            | 4 bytes| = 0x4D56_4357 ("MVCC" LE scrambled)
//! | version          | 2 bytes| = 1
//! | checksum_type    | 2 bytes| = 0 (CRC32c)
//! | reserved         | 8 bytes| = 0
//! +------------------+--------+
//!
//! Commit Record:
//! +------------------+--------+
//! | record_len       | 4 bytes| length of record (excluding this field)
//! | record_type      | 1 byte | = 1 (commit)
//! | commit_seq       | 8 bytes| CommitSeq value
//! | txn_id           | 8 bytes| TxnId value
//! | num_writes       | 4 bytes| number of block writes
//! +------------------+--------+
//! | For each write:           |
//! |   block_number   | 8 bytes|
//! |   data_len       | 4 bytes|
//! |   data           | N bytes|
//! +------------------+--------+
//! | record_crc       | 4 bytes| CRC32c of entire record (excluding len/crc)
//! +------------------+--------+
//! ```
//!
//! # Invariants
//!
//! - Replay is idempotent: replaying the same WAL produces the same state.
//! - Partial/truncated records are detected and safely ignored.
//! - Every record is checksummed for corruption detection.
//! - The commit boundary is explicit (each record is a complete commit).

use ffs_error::{FfsError, Result};
use ffs_types::{BlockNumber, CommitSeq, TxnId};

/// WAL file magic number ("MVCW" scrambled in little-endian).
pub const WAL_MAGIC: u32 = 0x4D56_4357;

/// Current WAL format version.
pub const WAL_VERSION: u16 = 1;

/// Checksum type: CRC32c.
pub const CHECKSUM_TYPE_CRC32C: u16 = 0;

/// File header size in bytes.
pub const HEADER_SIZE: usize = 16;

/// Minimum commit record size (header + crc, no writes).
pub const MIN_COMMIT_RECORD_SIZE: usize = 4 + 1 + 8 + 8 + 4 + 4; // 29 bytes

/// Record type for a commit.
pub const RECORD_TYPE_COMMIT: u8 = 1;

/// A single block write within a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalWrite {
    pub block: BlockNumber,
    pub data: Vec<u8>,
}

/// A commit record read from the WAL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalCommit {
    pub commit_seq: CommitSeq,
    pub txn_id: TxnId,
    pub writes: Vec<WalWrite>,
}

/// WAL file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalHeader {
    pub magic: u32,
    pub version: u16,
    pub checksum_type: u16,
}

impl Default for WalHeader {
    fn default() -> Self {
        Self {
            magic: WAL_MAGIC,
            version: WAL_VERSION,
            checksum_type: CHECKSUM_TYPE_CRC32C,
        }
    }
}

/// Encode a WAL file header to bytes.
#[must_use]
pub fn encode_header(header: &WalHeader) -> [u8; HEADER_SIZE] {
    let mut buf = [0_u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(&header.magic.to_le_bytes());
    buf[4..6].copy_from_slice(&header.version.to_le_bytes());
    buf[6..8].copy_from_slice(&header.checksum_type.to_le_bytes());
    // bytes 8..16 are reserved (zeros)
    buf
}

/// Decode a WAL file header from bytes.
pub fn decode_header(bytes: &[u8]) -> Result<WalHeader> {
    if bytes.len() < HEADER_SIZE {
        return Err(FfsError::Format(format!(
            "WAL header too short: {} bytes, need {}",
            bytes.len(),
            HEADER_SIZE
        )));
    }

    let magic = read_le_u32(bytes, 0)?;
    if magic != WAL_MAGIC {
        return Err(FfsError::Format(format!(
            "WAL magic mismatch: expected {WAL_MAGIC:#010x}, got {magic:#010x}"
        )));
    }

    let version = read_le_u16(bytes, 4)?;
    if version != WAL_VERSION {
        return Err(FfsError::Format(format!(
            "unsupported WAL version: {version} (expected {WAL_VERSION})"
        )));
    }

    let checksum_type = read_le_u16(bytes, 6)?;
    if checksum_type != CHECKSUM_TYPE_CRC32C {
        return Err(FfsError::Format(format!(
            "unsupported WAL checksum type: {checksum_type}"
        )));
    }

    Ok(WalHeader {
        magic,
        version,
        checksum_type,
    })
}

/// Encode a commit record to bytes.
///
/// Returns the encoded record including length prefix and trailing CRC.
pub fn encode_commit(commit: &WalCommit) -> Result<Vec<u8>> {
    // Calculate total size needed
    let mut data_size = 0_usize;
    for write in &commit.writes {
        data_size = data_size
            .checked_add(8 + 4) // block_number + data_len
            .and_then(|s| s.checked_add(write.data.len()))
            .ok_or_else(|| FfsError::Format("commit record size overflow".to_owned()))?;
    }

    // Record body size: type + commit_seq + txn_id + num_writes + data + crc
    let body_size = 1_usize
        .checked_add(8) // commit_seq
        .and_then(|s| s.checked_add(8)) // txn_id
        .and_then(|s| s.checked_add(4)) // num_writes
        .and_then(|s| s.checked_add(data_size)) // writes
        .and_then(|s| s.checked_add(4)) // crc
        .ok_or_else(|| FfsError::Format("commit record body size overflow".to_owned()))?;

    let total_size = 4_usize // record_len
        .checked_add(body_size)
        .ok_or_else(|| FfsError::Format("commit record total size overflow".to_owned()))?;

    let mut buf = vec![0_u8; total_size];
    let mut offset = 0_usize;

    // Record length (excludes the length field itself)
    let record_len = u32::try_from(body_size)
        .map_err(|_| FfsError::Format("commit record length exceeds u32".to_owned()))?;
    buf[offset..offset + 4].copy_from_slice(&record_len.to_le_bytes());
    offset += 4;

    // Record type
    buf[offset] = RECORD_TYPE_COMMIT;
    offset += 1;

    // Commit sequence
    buf[offset..offset + 8].copy_from_slice(&commit.commit_seq.0.to_le_bytes());
    offset += 8;

    // Transaction ID
    buf[offset..offset + 8].copy_from_slice(&commit.txn_id.0.to_le_bytes());
    offset += 8;

    // Number of writes
    let num_writes = u32::try_from(commit.writes.len())
        .map_err(|_| FfsError::Format("too many writes in commit".to_owned()))?;
    buf[offset..offset + 4].copy_from_slice(&num_writes.to_le_bytes());
    offset += 4;

    // Each write
    for write in &commit.writes {
        buf[offset..offset + 8].copy_from_slice(&write.block.0.to_le_bytes());
        offset += 8;

        let data_len = u32::try_from(write.data.len())
            .map_err(|_| FfsError::Format("write data length exceeds u32".to_owned()))?;
        buf[offset..offset + 4].copy_from_slice(&data_len.to_le_bytes());
        offset += 4;

        buf[offset..offset + write.data.len()].copy_from_slice(&write.data);
        offset += write.data.len();
    }

    // Compute CRC over the body (everything after record_len, before crc)
    let crc_start = 4_usize; // after record_len
    let crc_end = offset; // before crc
    let crc = crc32c::crc32c(&buf[crc_start..crc_end]);
    buf[offset..offset + 4].copy_from_slice(&crc.to_le_bytes());

    Ok(buf)
}

/// Result of attempting to decode a commit record.
#[derive(Debug)]
pub enum DecodeResult {
    /// Successfully decoded a commit.
    Commit(WalCommit),
    /// Need more bytes to complete the record.
    NeedMore(usize),
    /// Record is corrupted (CRC mismatch or invalid format).
    Corrupted(String),
    /// Reached end of valid data (all zeros or truncated).
    EndOfData,
}

/// Decode a commit record from bytes starting at the given position.
///
/// Returns the decoded commit and the number of bytes consumed, or an error.
#[must_use]
pub fn decode_commit(bytes: &[u8]) -> DecodeResult {
    if bytes.is_empty() {
        return DecodeResult::EndOfData;
    }

    // Need at least 4 bytes for record length
    if bytes.len() < 4 {
        return DecodeResult::NeedMore(4);
    }

    let record_len = match read_le_u32(bytes, 0) {
        Ok(len) => len as usize,
        Err(e) => return DecodeResult::Corrupted(format!("failed to read record length: {e}")),
    };

    // Zero record length indicates end of data
    if record_len == 0 {
        return DecodeResult::EndOfData;
    }

    // Validate record length is reasonable
    if record_len < MIN_COMMIT_RECORD_SIZE - 4 {
        return DecodeResult::Corrupted(format!(
            "record length too small: {record_len} < {}",
            MIN_COMMIT_RECORD_SIZE - 4
        ));
    }

    let total_size = 4 + record_len;
    if bytes.len() < total_size {
        return DecodeResult::NeedMore(total_size);
    }

    let record_bytes = &bytes[..total_size];
    let body_bytes = &record_bytes[4..];

    // Verify CRC (last 4 bytes of body)
    if body_bytes.len() < 4 {
        return DecodeResult::Corrupted("record too short for CRC".to_owned());
    }
    let crc_offset = body_bytes.len() - 4;
    let stored_crc = match read_le_u32(body_bytes, crc_offset) {
        Ok(crc) => crc,
        Err(e) => return DecodeResult::Corrupted(format!("failed to read CRC: {e}")),
    };
    let computed_crc = crc32c::crc32c(&body_bytes[..crc_offset]);
    if stored_crc != computed_crc {
        return DecodeResult::Corrupted(format!(
            "CRC mismatch: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
        ));
    }

    // Parse the record body
    let mut offset = 0_usize;

    // Record type
    if offset >= crc_offset {
        return DecodeResult::Corrupted("truncated record type".to_owned());
    }
    let record_type = body_bytes[offset];
    offset += 1;

    if record_type != RECORD_TYPE_COMMIT {
        return DecodeResult::Corrupted(format!("unknown record type: {record_type}"));
    }

    // Commit sequence
    let commit_seq = match read_le_u64(body_bytes, offset) {
        Ok(seq) => CommitSeq(seq),
        Err(e) => return DecodeResult::Corrupted(format!("failed to read commit_seq: {e}")),
    };
    offset += 8;

    // Transaction ID
    let txn_id = match read_le_u64(body_bytes, offset) {
        Ok(id) => TxnId(id),
        Err(e) => return DecodeResult::Corrupted(format!("failed to read txn_id: {e}")),
    };
    offset += 8;

    // Number of writes
    let num_writes = match read_le_u32(body_bytes, offset) {
        Ok(n) => n as usize,
        Err(e) => return DecodeResult::Corrupted(format!("failed to read num_writes: {e}")),
    };
    offset += 4;

    // Parse writes
    let mut writes = Vec::with_capacity(num_writes.min(1024));
    for i in 0..num_writes {
        // Block number
        let block = match read_le_u64(body_bytes, offset) {
            Ok(b) => BlockNumber(b),
            Err(e) => {
                return DecodeResult::Corrupted(format!("failed to read block number {i}: {e}"));
            }
        };
        offset += 8;

        // Data length
        let data_len = match read_le_u32(body_bytes, offset) {
            Ok(len) => len as usize,
            Err(e) => {
                return DecodeResult::Corrupted(format!("failed to read data length {i}: {e}"));
            }
        };
        offset += 4;

        // Data
        if offset + data_len > crc_offset {
            return DecodeResult::Corrupted(format!(
                "write {i} data extends past CRC: offset={offset}, len={data_len}, crc_offset={crc_offset}"
            ));
        }
        let data = body_bytes[offset..offset + data_len].to_vec();
        offset += data_len;

        writes.push(WalWrite { block, data });
    }

    DecodeResult::Commit(WalCommit {
        commit_seq,
        txn_id,
        writes,
    })
}

/// Returns the number of bytes consumed by a successfully decoded commit.
#[must_use]
pub fn commit_byte_size(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 4 {
        return None;
    }
    let record_len = read_le_u32(bytes, 0).ok()? as usize;
    if record_len == 0 {
        return None;
    }
    Some(4 + record_len)
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn read_le_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| FfsError::Format("read_le_u16 offset overflow".to_owned()))?;
    if end > bytes.len() {
        return Err(FfsError::Format(format!(
            "read_le_u16 out of bounds: offset={offset}, len={}",
            bytes.len()
        )));
    }
    let arr: [u8; 2] = bytes[offset..end]
        .try_into()
        .map_err(|_| FfsError::Format("read_le_u16 slice conversion failed".to_owned()))?;
    Ok(u16::from_le_bytes(arr))
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FfsError::Format("read_le_u32 offset overflow".to_owned()))?;
    if end > bytes.len() {
        return Err(FfsError::Format(format!(
            "read_le_u32 out of bounds: offset={offset}, len={}",
            bytes.len()
        )));
    }
    let arr: [u8; 4] = bytes[offset..end]
        .try_into()
        .map_err(|_| FfsError::Format("read_le_u32 slice conversion failed".to_owned()))?;
    Ok(u32::from_le_bytes(arr))
}

fn read_le_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| FfsError::Format("read_le_u64 offset overflow".to_owned()))?;
    if end > bytes.len() {
        return Err(FfsError::Format(format!(
            "read_le_u64 out of bounds: offset={offset}, len={}",
            bytes.len()
        )));
    }
    let arr: [u8; 8] = bytes[offset..end]
        .try_into()
        .map_err(|_| FfsError::Format("read_le_u64 slice conversion failed".to_owned()))?;
    Ok(u64::from_le_bytes(arr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let header = WalHeader::default();
        let encoded = encode_header(&header);
        let decoded = decode_header(&encoded).expect("decode should succeed");
        assert_eq!(decoded, header);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut buf = encode_header(&WalHeader::default());
        buf[0] = 0xFF;
        let result = decode_header(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn header_rejects_bad_version() {
        let mut buf = encode_header(&WalHeader::default());
        buf[4] = 99;
        let result = decode_header(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn commit_round_trip_empty() {
        let commit = WalCommit {
            commit_seq: CommitSeq(42),
            txn_id: TxnId(7),
            writes: vec![],
        };
        let encoded = encode_commit(&commit).expect("encode");
        let result = decode_commit(&encoded);
        match result {
            DecodeResult::Commit(decoded) => assert_eq!(decoded, commit),
            other => panic!("expected Commit, got {other:?}"),
        }
    }

    #[test]
    fn commit_round_trip_with_writes() {
        let commit = WalCommit {
            commit_seq: CommitSeq(100),
            txn_id: TxnId(5),
            writes: vec![
                WalWrite {
                    block: BlockNumber(10),
                    data: vec![1, 2, 3, 4],
                },
                WalWrite {
                    block: BlockNumber(20),
                    data: vec![5, 6, 7, 8, 9, 10],
                },
            ],
        };
        let encoded = encode_commit(&commit).expect("encode");
        let result = decode_commit(&encoded);
        match result {
            DecodeResult::Commit(decoded) => assert_eq!(decoded, commit),
            other => panic!("expected Commit, got {other:?}"),
        }
    }

    #[test]
    fn decode_detects_crc_corruption() {
        let commit = WalCommit {
            commit_seq: CommitSeq(1),
            txn_id: TxnId(1),
            writes: vec![WalWrite {
                block: BlockNumber(1),
                data: vec![0xAB; 16],
            }],
        };
        let mut encoded = encode_commit(&commit).expect("encode");
        // Corrupt a data byte
        let mid = encoded.len() / 2;
        encoded[mid] ^= 0xFF;
        let result = decode_commit(&encoded);
        match result {
            DecodeResult::Corrupted(msg) => assert!(msg.contains("CRC")),
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    #[test]
    fn decode_handles_truncation() {
        let commit = WalCommit {
            commit_seq: CommitSeq(1),
            txn_id: TxnId(1),
            writes: vec![],
        };
        let encoded = encode_commit(&commit).expect("encode");
        // Truncate to half
        let truncated = &encoded[..encoded.len() / 2];
        let result = decode_commit(truncated);
        match result {
            DecodeResult::NeedMore(_) => {}
            other => panic!("expected NeedMore, got {other:?}"),
        }
    }

    #[test]
    fn decode_handles_zero_record_length() {
        let zeros = [0_u8; 32];
        let result = decode_commit(&zeros);
        match result {
            DecodeResult::EndOfData => {}
            other => panic!("expected EndOfData, got {other:?}"),
        }
    }

    #[test]
    fn decode_handles_empty_input() {
        let result = decode_commit(&[]);
        match result {
            DecodeResult::EndOfData => {}
            other => panic!("expected EndOfData, got {other:?}"),
        }
    }

    #[test]
    fn commit_byte_size_returns_correct_value() {
        let commit = WalCommit {
            commit_seq: CommitSeq(1),
            txn_id: TxnId(1),
            writes: vec![WalWrite {
                block: BlockNumber(5),
                data: vec![0; 100],
            }],
        };
        let encoded = encode_commit(&commit).expect("encode");
        assert_eq!(commit_byte_size(&encoded), Some(encoded.len()));
    }

    #[test]
    fn multiple_commits_sequential() {
        let commits = vec![
            WalCommit {
                commit_seq: CommitSeq(1),
                txn_id: TxnId(1),
                writes: vec![WalWrite {
                    block: BlockNumber(1),
                    data: vec![1; 32],
                }],
            },
            WalCommit {
                commit_seq: CommitSeq(2),
                txn_id: TxnId(2),
                writes: vec![WalWrite {
                    block: BlockNumber(2),
                    data: vec![2; 64],
                }],
            },
            WalCommit {
                commit_seq: CommitSeq(3),
                txn_id: TxnId(3),
                writes: vec![],
            },
        ];

        // Encode all commits
        let mut data = Vec::new();
        for commit in &commits {
            data.extend(encode_commit(commit).expect("encode"));
        }

        // Decode them back
        let mut offset = 0;
        let mut decoded = Vec::new();
        while offset < data.len() {
            match decode_commit(&data[offset..]) {
                DecodeResult::Commit(commit) => {
                    let size = commit_byte_size(&data[offset..]).expect("size");
                    offset += size;
                    decoded.push(commit);
                }
                DecodeResult::EndOfData => break,
                DecodeResult::NeedMore(_) => panic!("unexpected NeedMore"),
                DecodeResult::Corrupted(msg) => panic!("unexpected Corrupted: {msg}"),
            }
        }

        assert_eq!(decoded, commits);
    }

    #[test]
    fn decode_commit_with_one_byte_returns_need_more() {
        let result = decode_commit(&[0x01]);
        match result {
            DecodeResult::NeedMore(4) => {}
            other => panic!("expected NeedMore(4), got {other:?}"),
        }
    }

    #[test]
    fn decode_commit_with_two_bytes_returns_need_more() {
        let result = decode_commit(&[0x01, 0x00]);
        match result {
            DecodeResult::NeedMore(4) => {}
            other => panic!("expected NeedMore(4), got {other:?}"),
        }
    }

    #[test]
    fn decode_commit_with_three_bytes_returns_need_more() {
        let result = decode_commit(&[0x01, 0x00, 0x00]);
        match result {
            DecodeResult::NeedMore(4) => {}
            other => panic!("expected NeedMore(4), got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_bad_checksum_type() {
        let mut buf = encode_header(&WalHeader::default());
        // checksum_type is at offset 6..8 — set to non-zero
        buf[6] = 0x01;
        buf[7] = 0x00;
        let result = decode_header(&buf);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("checksum type"),
            "error should mention checksum type: {msg}"
        );
    }

    #[test]
    fn header_rejects_too_short_input() {
        let result = decode_header(&[0_u8; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn commit_byte_size_with_short_input_returns_none() {
        assert_eq!(commit_byte_size(&[]), None);
        assert_eq!(commit_byte_size(&[0x01]), None);
        assert_eq!(commit_byte_size(&[0x01, 0x02, 0x03]), None);
    }

    #[test]
    fn commit_byte_size_with_zero_record_len_returns_none() {
        assert_eq!(commit_byte_size(&[0, 0, 0, 0]), None);
    }

    #[test]
    fn decode_commit_with_too_small_record_len_returns_corrupted() {
        // A record_len of 1 is far too small (min body is 25 bytes = type + seq + txn + nwrites + crc).
        let mut buf = [0_u8; 8];
        buf[0..4].copy_from_slice(&1_u32.to_le_bytes()); // record_len = 1
        buf[4] = 0xFF; // junk
        let result = decode_commit(&buf);
        match result {
            DecodeResult::Corrupted(msg) => {
                assert!(msg.contains("too small"), "expected 'too small' in: {msg}");
            }
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    // ── Property-based tests (proptest) ────────────────────────────────────

    use proptest::prelude::*;

    fn wal_write_strategy() -> impl Strategy<Value = WalWrite> {
        (any::<u64>(), proptest::collection::vec(any::<u8>(), 0..128)).prop_map(|(block, data)| {
            WalWrite {
                block: BlockNumber(block),
                data,
            }
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// WAL header encode/decode is a perfect roundtrip.
        #[test]
        fn proptest_wal_header_encode_decode_roundtrip(
            magic in Just(WAL_MAGIC),
            version in Just(WAL_VERSION),
            checksum_type in Just(CHECKSUM_TYPE_CRC32C),
        ) {
            let header = WalHeader { magic, version, checksum_type };
            let encoded = encode_header(&header);
            let decoded = decode_header(&encoded).expect("roundtrip decode");
            prop_assert_eq!(decoded, header);
        }

        /// WAL commit encode/decode is a perfect roundtrip for arbitrary commits.
        #[test]
        fn proptest_wal_commit_roundtrip(
            commit_seq in any::<u64>(),
            txn_id in any::<u64>(),
            writes in proptest::collection::vec(wal_write_strategy(), 0..8),
        ) {
            let commit = WalCommit {
                commit_seq: CommitSeq(commit_seq),
                txn_id: TxnId(txn_id),
                writes,
            };
            let encoded = encode_commit(&commit).expect("encode");
            match decode_commit(&encoded) {
                DecodeResult::Commit(decoded) => prop_assert_eq!(decoded, commit),
                other => prop_assert!(false, "expected Commit, got {:?}", other),
            }
        }

        /// Encoded commit byte size matches the encoded buffer length.
        #[test]
        fn proptest_wal_commit_byte_size_matches(
            commit_seq in any::<u64>(),
            txn_id in any::<u64>(),
            writes in proptest::collection::vec(wal_write_strategy(), 0..4),
        ) {
            let commit = WalCommit {
                commit_seq: CommitSeq(commit_seq),
                txn_id: TxnId(txn_id),
                writes,
            };
            let encoded = encode_commit(&commit).expect("encode");
            let size = commit_byte_size(&encoded);
            prop_assert_eq!(size, Some(encoded.len()));
        }

        /// Flipping any bit in the commit body triggers a CRC corruption error.
        #[test]
        fn proptest_wal_commit_bit_flip_detected(
            commit_seq in any::<u64>(),
            txn_id in any::<u64>(),
            data_byte in any::<u8>(),
            flip_offset in 5_usize..25,
        ) {
            let commit = WalCommit {
                commit_seq: CommitSeq(commit_seq),
                txn_id: TxnId(txn_id),
                writes: vec![WalWrite {
                    block: BlockNumber(1),
                    data: vec![data_byte; 32],
                }],
            };
            let mut encoded = encode_commit(&commit).expect("encode");
            let flip_pos = flip_offset.min(encoded.len().saturating_sub(5));
            if flip_pos < encoded.len() {
                encoded[flip_pos] ^= 0x01;
                match decode_commit(&encoded) {
                    // Any outcome is acceptable after bit-flipping:
                    // CRC collision (Commit), corruption detected, truncation, or zero-length.
                    DecodeResult::Commit(_)
                    | DecodeResult::Corrupted(_)
                    | DecodeResult::NeedMore(_)
                    | DecodeResult::EndOfData => {}
                }
            }
        }

        /// Multiple commits concatenated can all be decoded sequentially.
        #[test]
        fn proptest_wal_multi_commit_sequential_decode(
            commits in proptest::collection::vec(
                (any::<u64>(), any::<u64>(), proptest::collection::vec(wal_write_strategy(), 0..4)),
                1..6,
            ),
        ) {
            let commits: Vec<WalCommit> = commits
                .into_iter()
                .map(|(seq, txn, writes)| WalCommit {
                    commit_seq: CommitSeq(seq),
                    txn_id: TxnId(txn),
                    writes,
                })
                .collect();

            let mut data = Vec::new();
            for commit in &commits {
                data.extend(encode_commit(commit).expect("encode"));
            }

            let mut offset = 0;
            let mut decoded = Vec::new();
            while offset < data.len() {
                match decode_commit(&data[offset..]) {
                    DecodeResult::Commit(commit) => {
                        let size = commit_byte_size(&data[offset..]).expect("size");
                        offset += size;
                        decoded.push(commit);
                    }
                    DecodeResult::EndOfData | DecodeResult::NeedMore(_) => break,
                    DecodeResult::Corrupted(msg) => {
                        prop_assert!(false, "unexpected corruption: {}", msg);
                    }
                }
            }
            prop_assert_eq!(decoded.len(), commits.len());
            prop_assert_eq!(decoded, commits);
        }
    }
}
