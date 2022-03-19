// aarch64 neon
use core::arch::aarch64::{
    uint8x16_t, uint8x16x2_t, vaddq_u8, vandq_u8, vcgtq_u8, vcltq_u8, vdupq_n_u8, veorq_u8,
    vextq_u8, vld1q_u8, vld1q_u8_x2, vld2q_u8, vmaxvq_u8, vmovq_n_u8, vorrq_u8, vqsubq_u8,
    vqtbl1q_u8, vqtbl2q_u8, vshrq_n_u8, vsubq_u8,
};

// Map high nibble of "First Byte" to legal character length minus 1.
const FIRST_LENGTH_TABLE: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3];

const FIRST_RANGE_TABLE: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8];

const RANGE_MIN_TABLE: [u8; 16] = [
    0x00, 0x80, 0x80, 0x80, 0xA0, 0x80, 0x90, 0x80, 0xC2, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

const RANGE_MAX_TABLE: [u8; 16] = [
    0x7F, 0xBF, 0xBF, 0xBF, 0xBF, 0x9F, 0xBF, 0x8F, 0xF4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

const RANGE_ADJUST_TABLE: [u8; 32] = [
    2, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
];

#[cfg(target_feature = "neon")]
unsafe fn utf8_validate(mut data: *const u8, mut len: usize) -> bool {
    // TODO(@littledivy): Fallback to non simd STD
    assert!(len >= 32);

    let mut prev_input: uint8x16_t = vdupq_n_u8(0);
    let mut prev_first_len: uint8x16_t = vdupq_n_u8(0);

    let first_len_tbl: uint8x16_t = vld1q_u8(FIRST_LENGTH_TABLE.as_ptr());
    let first_range_tbl: uint8x16_t = vld1q_u8(FIRST_RANGE_TABLE.as_ptr());
    let range_min_tbl: uint8x16_t = vld1q_u8(RANGE_MIN_TABLE.as_ptr());
    let range_max_tbl: uint8x16_t = vld1q_u8(RANGE_MAX_TABLE.as_ptr());
    let range_adjust_tbl: uint8x16x2_t = vld2q_u8(RANGE_ADJUST_TABLE.as_ptr());

    let const_1: uint8x16_t = vdupq_n_u8(1);
    let const_2: uint8x16_t = vdupq_n_u8(2);
    let const_e0: uint8x16_t = vdupq_n_u8(0xe0);

    let mut error1 = vdupq_n_u8(0);
    let mut error2 = vdupq_n_u8(0);
    let mut error3 = vdupq_n_u8(0);
    let mut error4 = vdupq_n_u8(0);

    while len >= 32 {
        let input_pair: uint8x16x2_t = vld1q_u8_x2(data);
        let input_a = input_pair.0;
        let input_b = input_pair.1;

        let high_nibbles_a = vshrq_n_u8(input_a, 4);
        let high_nibbles_b = vshrq_n_u8(input_b, 4);

        let first_len_a = vqtbl1q_u8(first_len_tbl, high_nibbles_a);
        let first_len_b = vqtbl1q_u8(first_len_tbl, high_nibbles_b);

        let mut range_a = vorrq_u8(
            vqtbl1q_u8(first_range_tbl, first_len_a),
            vextq_u8(prev_first_len, first_len_a, 15),
        );
        let mut range_b = vorrq_u8(
            vqtbl1q_u8(first_range_tbl, first_len_b),
            vextq_u8(first_len_a, first_len_b, 15),
        );

        let mut tmp1_a: uint8x16_t;
        let mut tmp1_b: uint8x16_t;
        let mut tmp2_a: uint8x16_t;
        let mut tmp2_b: uint8x16_t;

        tmp1_a = vextq_u8(prev_first_len, first_len_a, 14);
        tmp1_a = vqsubq_u8(tmp1_a, const_1);
        range_a = vorrq_u8(range_a, tmp1_a);

        tmp1_b = vextq_u8(first_len_a, first_len_b, 14);
        tmp1_b = vqsubq_u8(tmp1_b, const_1);
        range_b = vorrq_u8(range_b, tmp1_b);

        tmp2_a = vextq_u8(prev_first_len, first_len_a, 13);
        tmp2_a = vqsubq_u8(tmp2_a, const_2);
        range_a = vorrq_u8(range_a, tmp2_a);

        tmp2_b = vextq_u8(first_len_a, first_len_b, 13);
        tmp2_b = vqsubq_u8(tmp2_b, const_2);
        range_b = vorrq_u8(range_b, tmp2_b);

        let shift1_a = vextq_u8(prev_input, input_a, 15);
        let pos_a = vsubq_u8(shift1_a, const_e0);
        range_a = vaddq_u8(range_a, vqtbl2q_u8(range_adjust_tbl, pos_a));

        let shift1_b = vextq_u8(input_a, input_b, 15);
        let pos_b = vsubq_u8(shift1_b, const_e0);
        range_b = vaddq_u8(range_b, vqtbl2q_u8(range_adjust_tbl, pos_b));

        let minv_a = vqtbl1q_u8(range_min_tbl, range_a);
        let maxv_a = vqtbl1q_u8(range_max_tbl, range_a);

        let minv_b = vqtbl1q_u8(range_min_tbl, range_b);
        let maxv_b = vqtbl1q_u8(range_max_tbl, range_b);

        error1 = vorrq_u8(error1, vcltq_u8(input_a, minv_a));
        error2 = vorrq_u8(error2, vcgtq_u8(input_a, maxv_a));

        error3 = vorrq_u8(error3, vcltq_u8(input_b, minv_b));
        error4 = vorrq_u8(error4, vcgtq_u8(input_b, maxv_b));

        prev_input = input_b;
        prev_first_len = first_len_b;

        data = data.add(32);
        len -= 32;
    }

    error1 = vorrq_u8(error1, error2);
    error1 = vorrq_u8(error1, error3);
    error1 = vorrq_u8(error1, error4);

    if vmaxvq_u8(error1) != 0 {
        return false;
    }

    // TODO
    true
}

#[cfg(not(target_feature = "neon"))]
compile_error!("Only aarch64 neon simd supported.");

fn main() {
  static TEST_DATA: &[u8] = include_bytes!("../test.txt");
  assert!(unsafe { utf8_validate(TEST_DATA.as_ptr(), TEST_DATA.len()) });
}
