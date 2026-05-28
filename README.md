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
};
```

## Usage

0.16.x Zig compiler is required

```bash
# build the compiler
zig build -Doptimize=ReleaseSafe -Dllvm
# convert code into IR and run the IR in a VM
./zig-out/bin/ion-stage0 eval
./zig-out/bin/ion-stage0 eval in.ion
# show the tokens before evaluating
./zig-out/bin/ion-stage0 eval in.ion --dump-tokens
# show the AST before evaluating
./zig-out/bin/ion-stage0 eval in.ion --dump-ast
# show the IR before evaluating
./zig-out/bin/ion-stage0 eval in.ion --dump-ir
# show what the VM is doing and print all globals
./zig-out/bin/ion-stage0 eval in.ion --dump-vm
# transpile code to zig
./zig-out/bin/ion-stage0 build in.ion out.zig
zig run out.zig
# transpile code to zig and run it using the zig compiler
./zig-out/bin/ion-stage0 build in.ion out.zig --run
# more about the cli usage
./zig-out/bin/ion-stage0 --help

# src/stage1/main.ion has some example ion code to use
./zig-out/bin/ion-stage0 build src/stage1/main.ion out.zig --run
```
