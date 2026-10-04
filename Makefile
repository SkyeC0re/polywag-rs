docs-katex:
	RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps

test-silent:
	RUSTFLAGS="-Awarnings" cargo test