/// This file is part of same and is dual-licensed under the MIT and Apache v2 licenses
/// See the LICENSE file for details
extern crate dft;

use self::dft::{Operation, Plan};
use std::f32::consts::PI;
use std::str::FromStr;
use std::fmt;

const ZERO_FREQ_HZ: f32 = 1562.5f32;  // Frequency of zero bit
const ONE_FREQ_HZ: f32 = 2083.3f32;   // Frequency of one bit
const MAX_FREQ_DELTA: f32 = 0.14f32;  // Maximum frequency deviation
const BIT_DURATION_S: f32 = 1.92e-3;  // Period of a bit (3 zero cycles/4 one cycles)

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Message {
    Header {
        originator: String,
        event: String,
        locations: Vec<u16>,
        purge_time: u16,
        day: u16,
        hour: u16,
        minute: u16,
        station: String,
    },
    EOM,
}

impl Message {
    pub fn new_header(org: &str, evt: &str, loc: &[u16], purge: u16, day: u16, hour: u16, minute: u16, station: &str) -> Message {
        let mut loc_v = Vec::with_capacity(loc.len());
        loc_v.extend_from_slice(loc);
        Message::Header {
            originator: String::from(org),
            event: String::from(evt),
            locations: loc_v,
            purge_time: purge,
            day: day,
            hour: hour,
            minute: minute,
            station: String::from(station),
        }
    }

    /// Parse a SAME message
    pub fn from_bytes(bytes: &[u8]) -> Option<Message> {
        const ATTN_CODE: [u8; 4] = [0x5a, 0x43, 0x5a, 0x43]; // ZCZC
        const EOM_CODE: [u8; 4] = [0x4e, 0x4e, 0x4e, 0x4e]; // NNNN
        let mut buffer = String::with_capacity(10);

        let mut originator = String::with_capacity(3);
        let mut event = String::with_capacity(3);
        let mut locations: Vec<u16> = Vec::new();

        let mut found_purge = false;
        let mut purge_time = 0u16;
        let mut datestr = String::with_capacity(7);
        let mut station = String::with_capacity(7);

        // Find the attention code
        let mut msg_start = 0;
        while msg_start + 4 < bytes.len() && bytes[msg_start] != 0x5a && bytes[msg_start] != 0x4e {
            msg_start += 1;
        }
        if msg_start >= bytes.len() {
            return None;
        }
        // Check for EOM
        let mut is_eom = true;
        for i in msg_start..msg_start + 4 {
            if bytes[i] != EOM_CODE[i - msg_start] {
                is_eom = false;
                break;
            }
        }
        if is_eom {
            return Some(Message::EOM);
        }

        // Failing that, check for attn. code
        for i in msg_start..msg_start + 4 {
            if bytes[i] != ATTN_CODE[i - msg_start] {
                return None;
            }
        }

        for byte in bytes {
            let ch = *byte as char;
            if *byte == 0xab {
                continue;
            }
            if buffer == "ZCZC" {
                buffer = String::new();
            } else if ch == '+' {
                if let Ok(location) = u16::from_str(&buffer) {
                    locations.push(location);
                }
                found_purge = true;
                buffer = String::new();
            } else if ch == '-' && buffer.len() > 0 {
                // Flush the buffer
                if originator.len() == 0 {
                    originator = buffer;
                } else if event.len() == 0 {
                    event = buffer;
                } else if !found_purge {
                    if let Ok(location) = u16::from_str(&buffer) {
                        locations.push(location);
                    } else {
                        return None;
                    }
                } else {
                    if purge_time == 0 {
                        if let Ok(time) = u16::from_str(&buffer) {
                            purge_time = time;
                        } else {
                            return None;
                        }
                    } else if datestr.len() == 0 {
                        datestr = buffer;
                    } else {
                        station = buffer;
                        break;
                    }
                }
                buffer = String::new();
            } else {
                buffer.push(ch);
            }
        }
        let mut dayofyear = 1;
        let mut hour = 0;
        buffer = String::new();
        for (i, c) in datestr.chars().enumerate() {
            if i == 3 {
                dayofyear = match u16::from_str(&buffer) {
                    Ok(d) => d,
                    _ => 1,
                };
                buffer = String::new()
            } else if i == 5 {
                hour = match u16::from_str(&buffer) {
                    Ok(h) => h,
                    _ => 0,
                };
                buffer = String::new()
            }
            buffer.push(c);
        }
        let minute = match u16::from_str(&buffer) {
            Ok(m) => m,
            _ => 0,
        };

        Some(Message::Header {
                 originator: originator,
                 event: event,
                 locations: locations,
                 purge_time: purge_time,
                 day: dayofyear,
                 hour: hour,
                 minute: minute,
                 station: station,
             })
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Message::Header {
                 ref originator,
                 ref event,
                 ref locations,
                 ref purge_time,
                 ref day,
                 ref hour,
                 ref minute,
                 ref station,
             } => {
                let mut loc_str = String::new();
                for l in locations {
                    loc_str.push_str(&format!("-{:06}", l));
                }
                write!(f,
                       "ZCZC-{}-{}{}+{:04}-{:03}{:02}{:02}-{}-",
                       originator,
                       event,
                       loc_str,
                       purge_time,
                       day,
                       hour,
                       minute,
                       station)
            }
            &Message::EOM => write!(f, "NNNN"),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Bit {
    Zero,
    One,
    Indeterminate,
}

struct BitIterator {
    signal: Vec<f32>,
    sample_rate_hz: f32,
    index: usize,
    dft_size: usize,
    window_size: usize,
    sample_offset: usize,
    decoding: bool,
}

impl BitIterator {
    fn new(samples: Vec<f32>, sample_rate_hz: f32, dft_size: usize) -> BitIterator {
        let window_size = (sample_rate_hz * BIT_DURATION_S) as usize;
        BitIterator {
            signal: samples,
            sample_rate_hz: sample_rate_hz,
            dft_size: dft_size,
            window_size: window_size,
            index: 0,
            sample_offset: 0,
            decoding: false,
        }
    }

    fn calculate_offset(&self, window_start: usize) -> usize {
        let mut offset = 0;
        let mut peak_offset_ampl = 0f32;
        for n in 0..self.window_size / 2 {
            if n > window_start {
                break;
            }
            let mut normalized_window = Vec::with_capacity(self.window_size);
            let mut w_max = 0f32;
            for i in 0..self.window_size {
                let w = self.signal[i + window_start - n];
                if w > w_max {
                    w_max = w;
                }
            }
            for i in 0..self.window_size {
                normalized_window.push(self.signal[i + window_start - n] / w_max);
            }
            let ft_data = real_dft(&blackman_harris_window(&normalized_window), self.dft_size);
            let freq_step = self.sample_rate_hz / (ft_data.len() as f32) / 2f32;
            let freq_range = freq_step;
            let mut peak_freq = 10000f32;
            let mut peak_ampl = 0f32;
            for n in 0..ft_data.len() / 2 {
                if n % 2 == 0 {
                    let point = (ft_data[n].powi(2) + ft_data[n + 1].powi(2)).sqrt();
                    let freq = (n as f32) * freq_step / 2f32;
                    if (ZERO_FREQ_HZ - freq).abs() < peak_freq {
                        peak_ampl = point;
                        peak_freq = (ZERO_FREQ_HZ - freq).abs();
                    }
                }
            }
            if peak_ampl > peak_offset_ampl {
                peak_offset_ampl = peak_ampl;
                offset = n;
            } else {
                //println!("Offset {} is worse ({:01.4} @ {:4.1} Hz)", n, peak_ampl, (ZERO_FREQ_HZ - peak_freq).abs());
            }
        }
        //println!("Optimum offset appears to be {} ({})", offset, window_start - offset);
        offset
    }
}

impl Iterator for BitIterator {
    type Item = Bit;

    fn next(&mut self) -> Option<Bit> {
        let start_time = (self.index as f32) * BIT_DURATION_S;
        let window_start = (start_time * self.sample_rate_hz) as usize - self.sample_offset;
        if window_start >= self.signal.len() {
            return None;
        }
        self.index += 1;
        let mut window: Vec<f32> = Vec::with_capacity(self.window_size);
        for n in 0..self.window_size {
            if n + window_start >= self.signal.len() {
                window.push(0f32);
            } else {
                window.push(self.signal[n + window_start]);
            }
        }
        let bit = decode_bit(&window, self.sample_rate_hz, self.dft_size);
        if !self.decoding && bit != Bit::Indeterminate {
            self.decoding = true;
            self.sample_offset = self.calculate_offset(window_start);
        } else if bit == Bit::Indeterminate {
            self.decoding = false;
            self.sample_offset = 0;
        }
        Some(bit)
    }
}

/// Phase accumulator for encoding phase-coherent FSK signals
struct PhaseAccumulator {
    phase: f32,
    delta_phase: f32,
    sample_rate_hz: f32,
}

impl PhaseAccumulator {
    pub fn new(freq_hz: f32, sample_rate_hz: f32) -> PhaseAccumulator {
        PhaseAccumulator {
            phase: 0f32,
            delta_phase: (freq_hz / sample_rate_hz),
            sample_rate_hz: sample_rate_hz,
        }
    }

    /// Set the frequency of the oscillator
    pub fn set_freq(&mut self, new_freq_hz: f32) {
        self.delta_phase = new_freq_hz / self.sample_rate_hz;
    }

    /// Get the next sine sample from the accumulator
    pub fn next_sample(&mut self) -> f32 {
        let sample = (2f32 * PI * self.phase).sin();
        self.phase += self.delta_phase;
        sample
    }
}

/// FSK-encode a byte for a SAME message
fn encode_byte(osc: &mut PhaseAccumulator, byte: &u8) -> Vec<f32> {
    let samples_per_bit = (BIT_DURATION_S * osc.sample_rate_hz).round() as usize;
    let mut samples = Vec::with_capacity(samples_per_bit * 8);
    // Encode one bit at a time (little-endian)
    for b in 0..8 {
        let mut s = Vec::with_capacity(samples_per_bit);
        let n_cycles = if byte & (1 << b) == 0 {
            osc.set_freq(ZERO_FREQ_HZ);
            6
        } else {
            osc.set_freq(ONE_FREQ_HZ);
            8
        };
        let mut n_zero_crossings = 0;
        let mut last_sample = 0f32;
        while n_zero_crossings < n_cycles {
            let next_sample = osc.next_sample();
            s.push(next_sample);
            if (last_sample < 0f32 && next_sample >= 0f32) || (last_sample > 0f32 && next_sample <= 0f32) {
                n_zero_crossings += 1;
            }
            last_sample = next_sample;
        }
        samples.append(&mut s);
    }
    samples
}

/// Encode a string of bytes into a SAME message
pub fn encode_bytes(bytes: &[u8], sample_rate_hz: f32) -> Vec<f32> {
    let mut osc = PhaseAccumulator::new(1f32, sample_rate_hz);
    let mut samples: Vec<f32> = Vec::new();
    for b in bytes {
        let mut s = encode_byte(&mut osc, b);
        samples.append(&mut s);
    }
    samples
}

/// Decode an FSK-encoded bit from a SAME message
fn decode_bit(window: &[f32], sample_rate_hz: f32, dft_size: usize) -> Bit {
    const BIT_THRESHOLD: f32 = 0.1f32;
    // Normalize the samples to -1, 1
    let mut normalized_window = Vec::with_capacity(window.len());
    let mut w_max = 0f32;
    for w in window {
        if *w > w_max {
            w_max = *w;
        }
    }
    if w_max < BIT_THRESHOLD {
        return Bit::Indeterminate;
    }
    for i in 0..window.len() {
        normalized_window.push(window[i] / w_max);
    }
    let ft_data = real_dft(&blackman_harris_window(&normalized_window), dft_size);
    let freq_step = sample_rate_hz / (ft_data.len() as f32) / 2f32;
    let freq_range = freq_step;
    let mut peak_freq = 0f32;
    let mut peak_ampl = 0f32;
    for n in 0..ft_data.len() / 2 {
        if n % 2 == 0 {
            let point = (ft_data[n].powi(2) + ft_data[n + 1].powi(2)).sqrt();
            let freq = (n as f32) * freq_step / 2f32;
            if freq > 0f32 && point > peak_ampl {
                peak_freq = freq;
                peak_ampl = point;
            }
        }
    }
    // Decode
    let zero_delta = MAX_FREQ_DELTA * ZERO_FREQ_HZ;
    let one_delta = MAX_FREQ_DELTA * ONE_FREQ_HZ;
    //println!("    {:01.4} @ {:4.1} Hz", peak_ampl, peak_freq);
    if peak_ampl < BIT_THRESHOLD {
        Bit::Indeterminate
    } else if ZERO_FREQ_HZ - zero_delta < peak_freq && peak_freq < ZERO_FREQ_HZ + zero_delta {
        Bit::Zero
    } else if ONE_FREQ_HZ - one_delta < peak_freq && peak_freq < ONE_FREQ_HZ + one_delta {
        Bit::One
    } else {
        Bit::Indeterminate
    }
}

/// Decode an FSK-encoded SAME signal into strings of bytes
pub fn decode_bytes(samples: Vec<f32>, sample_rate_hz: f32, dft_size: usize) -> Vec<Vec<u8>> {
    const PREAMBLE: [Bit; 8] = [Bit::One, Bit::One, Bit::Zero, Bit::One, Bit::Zero, Bit::One,
                                Bit::Zero, Bit::One];
    let mut bytestrings = Vec::new();
    let mut decoded_bytes = Vec::new();
    let mut curr_byte = 0u8;
    let mut curr_byte_pos = 0u8;
    let mut decoding = false;
    let mut last_byte: [Bit; 8] = [Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate,
                                   Bit::Indeterminate];
    let bits = BitIterator::new(samples, sample_rate_hz, dft_size);
    for bit in bits {
        if !decoding {
            for i in 0..7 {
                last_byte[i] = last_byte[i + 1]
            }
            last_byte[7] = bit;
            if last_byte == PREAMBLE {
                // Start decoding when we detect a preamble byte
                decoding = true;
                curr_byte = 0xab;
                curr_byte_pos = 7;
            }
        }
        if decoding {
            match bit {
                Bit::Zero => {
                    // Do nothing
                }
                Bit::One => {
                    curr_byte |= 1 << curr_byte_pos;
                }
                Bit::Indeterminate => {
                    // Message ended
                    decoding = false;
                    bytestrings.push(decoded_bytes);
                    decoded_bytes = Vec::new();
                }
            }
            curr_byte_pos = (curr_byte_pos + 1) % 8;
            if curr_byte_pos == 0 {
                decoded_bytes.push(curr_byte);
                curr_byte = 0;
            }
        }
    }
    bytestrings
}

/// Apply a Blackman-Harris window to a discrete-time signal
fn blackman_harris_window(samples: &[f32]) -> Vec<f32> {
    const A0: f32 = 0.35875;
    const A1: f32 = 0.48829;
    const A2: f32 = 0.14128;
    const A3: f32 = 0.01168;
    const TWO_PI: f32 = PI * 2f32;
    const FOUR_PI: f32 = PI * 4f32;
    const SIX_PI: f32 = PI * 6f32;

    let mut windowed = Vec::with_capacity(samples.len());
    let denom = 1f32 / (samples.len() - 1) as f32;
    for n in 0..samples.len() {
        let one_term = A1 * (TWO_PI * (n as f32) * denom).cos();
        let two_term = A2 * (FOUR_PI * (n as f32) * denom).cos();
        let three_term = A3 * (SIX_PI * (n as f32) * denom).cos();
        let weight = A0 - one_term + two_term - three_term;
        windowed.push(weight * samples[n]);
    }
    windowed
}

/// Calculate the DFT of a real-valued signal
fn real_dft(samples: &[f32], size: usize) -> Vec<f32> {
    let mut ft_data = Vec::with_capacity(samples.len());
    for n in 0..size {
        if n < samples.len() {
            ft_data.push(samples[n]);
        } else {
            ft_data.push(0f32);
        }
    }
    let plan = Plan::new(Operation::Forward, size);
    dft::transform(&mut ft_data, &plan);

    let mut freq_data = Vec::with_capacity(samples.len() / 2 + 1);
    for i in 0..((ft_data.len() / 2) + 1) {
        freq_data.push(ft_data[i] / (samples.len() as f32));
    }
    freq_data
}

#[cfg(test)]
mod tests {
    extern crate rand;
    use self::rand::distributions::{IndependentSample, Range};

    use super::*;

    const DFT_SIZE: usize = 512;
    const SAMPLE_RATE_HZ: f32 = 44_100f32;

    #[test]
    pub fn test_decode_32() {
        let mut bytes: Vec<u8> = Vec::with_capacity(32);
        for i in 0..32 {
            bytes.push(i as u8);
        }
        bytes[0] = 0xab;
        let samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let len = samples.len();

        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);
        assert_eq!(decoded_bytes.len(), 1);
        assert_eq!(decoded_bytes[0].len(), bytes.len());
        for n in 0..decoded_bytes.len() {
            assert_eq!(bytes[n], decoded_bytes[0][n]);
        }
    }

    #[test]
    pub fn test_decode_64() {
        let mut bytes: Vec<u8> = Vec::with_capacity(64);
        for i in 0..64 {
            bytes.push(i as u8);
        }
        bytes[0] = 0xab;
        let samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let len = samples.len();

        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);
        assert_eq!(decoded_bytes.len(), 1);
        assert_eq!(decoded_bytes[0].len(), bytes.len());
        for n in 0..decoded_bytes.len() {
            assert_eq!(bytes[n], decoded_bytes[0][n]);
        }
    }

    #[test]
    pub fn test_decode_128() {
        let mut bytes: Vec<u8> = Vec::with_capacity(128);
        for i in 0..128 {
            bytes.push(i as u8);
        }
        bytes[0] = 0xab;
        let samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let len = samples.len();

        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);
        assert_eq!(decoded_bytes.len(), 1);
        assert_eq!(decoded_bytes[0].len(), bytes.len());
        for n in 0..decoded_bytes.len() {
            assert_eq!(bytes[n], decoded_bytes[0][n]);
        }
    }

    #[test]
    pub fn test_decode_256() {
        let mut bytes: Vec<u8> = Vec::with_capacity(256);
        for i in 0..256 {
            bytes.push(i as u8);
        }
        bytes[0] = 0xab;
        let samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let len = samples.len();

        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);
        assert_eq!(decoded_bytes.len(), 1);
        assert_eq!(decoded_bytes[0].len(), bytes.len());
        for n in 0..decoded_bytes.len() {
            assert_eq!(bytes[n], decoded_bytes[0][n]);
        }
    }

    #[test]
    pub fn test_decode_invalid() {
        let mut bytes: Vec<u8> = Vec::with_capacity(256);
        for i in 0..256 {
            bytes.push(0);
            bytes.push(0xff);
        }
        let samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);
        assert_eq!(decoded_bytes.len(), 0);
    }

    #[test]
    pub fn test_decode_with_noise() {
        const NOISE_AMPLITUDE: f32 = 0.7;
        let mut bytes: Vec<u8> = Vec::with_capacity(256);
        for i in 0..256 {
            bytes.push(i as u8);
        }
        bytes[0] = 0xab;
        let mut samples = encode_bytes(&bytes, SAMPLE_RATE_HZ);
        let len = samples.len();

        // Add some zero-mean noise
        let mut rng = rand::thread_rng();
        let noise = Range::new(0f32, 1f32);
        for i in 0..len {
            samples[i] = samples[i] + NOISE_AMPLITUDE * noise.ind_sample(&mut rng);
        }

        let decoded_bytes = decode_bytes(samples, SAMPLE_RATE_HZ, DFT_SIZE);

        assert_eq!(decoded_bytes.len(), 1);
        assert_eq!(decoded_bytes[0].len(), bytes.len());
        for n in 0..decoded_bytes[0].len() {
            assert_eq!(bytes[n], decoded_bytes[0][n]);
        }
    }
}
