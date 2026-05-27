# ion

## Example code

```rust
// all functions are anonymous
let double = fn(num: isize): isize {
    // trailing semicolon can be omitted to return
    // from a scope just like in Rust
    num * 2
};

let four = 4;
let main = fn() {
    print("(1 + 4 * 6) * 2 = ");
    // int literals default to isize
    print(double(1 + four * 6));
    print("\n");
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
