let
  # Pinned nixpkgs, deterministic. Last updated: 2026-01-05.
  pkgs = import (fetchTarball (
    "https://github.com/NixOS/nixpkgs/archive/c8ba0d39ac852f040726e33bb08c24a953934568.tar.gz"
  )) { };

  xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
in
pkgs.mkShell {
  XNNPACK_DIR = "${xnnpack}";
  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
  buildInputs = [
    xnnpack
    pkgs.llvmPackages.clang
    pkgs.cargo
    pkgs.rustc
    (pkgs.rustfmt.override { asNightly = true; })
  ];
}
