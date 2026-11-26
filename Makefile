docs-katex:
	RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps

test-silent:
	RUSTFLAGS="-Awarnings" cargo test

test-miri:
	MIRIFLAGS="-Zmiri-no-extra-rounding-error" cargo +nightly miri test