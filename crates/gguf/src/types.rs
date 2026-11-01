use anyhow::{bail, Result};

/// GGML tensor types (ids match ggml.h). Only the ones we can size are listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            _ => bail!("unsupported ggml type id {id}"),
        })
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (u64, u64) {
        use GgmlType::*;
        match self {
            F32 | I32 => (1, 4),
            F16 | BF16 | I16 => (1, 2),
            I8 => (1, 1),
            I64 | F64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            IQ4_NL => (32, 18),
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ2_S => (256, 82),
            IQ3_XXS => (256, 98),
            IQ3_S => (256, 110),
            IQ1_S => (256, 50),
            IQ1_M => (256, 56),
            IQ4_XS => (256, 136),
        }
    }

    pub fn row_bytes(self, n: u64) -> u64 {
        let (be, bb) = self.block();
        assert!(n % be == 0, "{self:?}: row of {n} not a multiple of block {be}");
        n / be * bb
    }

    pub fn bits_per_weight(self) -> f64 {
        let (be, bb) = self.block();
        bb as f64 * 8.0 / be as f64
    }
}
