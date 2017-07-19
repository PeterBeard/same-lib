# same-lib

`same-lib` is a Rust library for encoding and decoding audio data according to
the Specific Area Message Encoding (SAME) standard.

## Status

### Encoder

Working

### Decoder

Sort of works but not at all reliably. Won't decode its own output if it's too
long so there's definitely something wrong somewhere.
