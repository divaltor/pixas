# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust project named "pixas" using Rust 2024 edition. Currently contains a minimal "Hello, world!" application.

## Development Commands

- `cargo run` - Build and run the application
- `cargo build` - Build the project
- `cargo build --release` - Build optimized release version
- `cargo check` - Check code without building
- `cargo test` - Run tests
- `cargo fmt` - Format code
- `cargo clippy` - Run linter

## Architecture

Simple binary crate with entry point in `src/main.rs`.