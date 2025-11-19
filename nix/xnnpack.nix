{
  clangStdenv,
  fetchFromGitHub,
  cmake,
  python3,
}:
let
  cpuinfo_src = fetchFromGitHub {
    owner = "pytorch";
    repo = "cpuinfo";
    rev = "8a9210069b5a37dd89ed118a783945502a30a4ae";
    sha256 = "sha256-zEhgihmzfuTJkbIqtHuYv/d7F7j6QRJqU9ZHdIwW7bI=";
  };
  fxdiv_src = fetchFromGitHub {
    owner = "Maratyszcza";
    repo = "FXdiv";
    rev = "b408327ac2a15ec3e43352421954f5b1967701d1";
    sha256 = "sha256-BEjscsejYVhRxDAmah5DT3+bglp8G5wUTTYL7+HjWds=";
  };
  pthreadpool_src = fetchFromGitHub {
    owner = "google";
    repo = "pthreadpool";
    rev = "0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e";
    sha256 = "sha256-FPUKjWARY4TdUq7ni48tnszEdmVYxPXIgtddPBHn/U0=";
  };
  kleidiai_src = fetchFromGitHub {
    owner = "ARM-software";
    repo = "kleidiai";
    rev = "63205aa90afa6803d8f58bc3081b69288e9f1906";
    sha256 = "sha256-//o0/geqeGum6blRpMwDIcCRsYnm2/2illT0+FKxkZ0=";
  };
  gbenchmark_src = fetchFromGitHub {
    owner = "google";
    repo = "benchmark";
    rev = "8d4fdd6e6e003867045e0bb3473b5b423818e4b7";
    sha256 = "sha256-SdPmiHrpEQy8J0F7QOVb1dglhqG7O0RoLW//9UTaE0Q=";
  };
  gtest_src = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    rev = "35d0c365609296fa4730d62057c487e3cfa030ff";
    sha256 = "sha256-ulmZZ2qQePWMnPdYYcYqZSBsKdhKwr5cStTHGhSYK6M=";
  };
in

clangStdenv.mkDerivation {
  name = "XNNPACK";
  src = fetchFromGitHub {
    owner = "google";
    repo = "XNNPACK";
    rev = "3f83cc608b7a3ef63e59660c08970e3904ae50ae";
    sha256 = "sha256-8q6sWpZyDCHEAqEoY3yDz9GKTUZnfAz5xoQ9A/JSwck=";
  };
  nativeBuildInputs = [
    cmake
    python3

  ];
  cmakeFlags = [
    "-DCPUINFO_SOURCE_DIR=${cpuinfo_src}"
    "-DFXDIV_SOURCE_DIR=${fxdiv_src}"
    "-DPTHREADPOOL_SOURCE_DIR=${pthreadpool_src}"
    "-DKLEIDIAI_SOURCE_DIR=${kleidiai_src}"
    "-DGOOGLEBENCHMARK_SOURCE_DIR=${gbenchmark_src}"
    "-DGOOGLETEST_SOURCE_DIR=${gtest_src}"
    "-DXNNPACK_ENABLE_KLEIDIAI=ON"
  ];
}
