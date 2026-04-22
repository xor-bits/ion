# ion

## Example code

```rust
// All functions are anonymous, including extern functions.
let prints = fn(s: []u8) {
    // You can reference external symbols using this expression:
    let printf = extern fn(s: *u8, ..): c_int @ "printf";
    printf("%.*s\x00".ptr, s.len, s.ptr);
};

let printi = fn(i: u32) {
    // ... and since it is an expression, you don't need to save it.
    (extern fn(s: *u8, ..): c_int @ "printf")("%lu\x00".ptr, i);
};

let double = fn(num: u32): u32 {
    // Trailing semicolon can be omitted to return
    // from a scope just like in Rust.
    // `return` keyword is still WIP.
    num * 2
};

let four = 4;
let main = fn() {
    prints("(1 + 4 * 6) * 2 = ");
    printi(double(1 + four * 6));
    prints("\n");
};
```

## Usage

0.15.x Zig compiler is required

```bash
# build the transpiler
zig build
# run the transpiler (ion-stage0 <src> <output>)
./zig-out/bin/ion-stage0 ./src/stage1/main.ion out.zig
# run the generated Zig code
zig run -lc out.zig

# or everything in one command (-Ddump is optional)
zig build run -Ddump # -Ddump dumps the generated Zig code
```
