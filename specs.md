# voxcpm-rs

This is a package designed to make the consumption of the TTS model VoxCPM2 easy in rust.

It provides a vulkan (+ cpu as a fallback) implementation of the model so you can run it locally with the highest possible speed in your AMD cards (or NVIDIA, vulkan is very portable) and even directly on the CPU.
It uses burn as the intermediary for making the implementation dozens of times simpler.

Uses the reference official implementation in vendor.

The API is simple to use and convenient.
