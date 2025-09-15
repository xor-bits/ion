{
  description = "Nix devenv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        # `nix develop`
        devShells.default = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            pkg-config
            zig_0_15
            zls_0_15
            llvmPackages_21.llvm
            llvmPackages_21.clang
          ];
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        };
      }
    );
}
