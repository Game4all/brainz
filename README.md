<div align="center">
    <h1><code>brainz</code></h1>
    <i>Braaaaaaaaaaaaaiiiiiiiinnnnnnnnsssss 🧠🧟‍♂️</i>
    <br/>
    <i>A small tensor library with just enough operators for Deep Learning written in pure Zig</i>
    <hr>
</div>


# Installation and usage

1. Get the last version of the library by running:

```bash
zig fetch --save git+https://github.com/Game4all/brainz
```

2. Check the examples in `examples/` folder to get started

# Why ? 

- Because it's fun to reinvent the wheel?
- This library was created as an exercice to understand how the machinery that powers huge frameworks like TensorFlow, PyTorch and GGML and to get a small tensor framework to experiment with neural nets in pure zig.
- This library mainly targets CPUs enabling easy deployment everywhere at the cost of peak throughput and performance. If you're interested in using GPUs / squeezing out every single bit of performance out of your hardware, you may be interested in checking out [zigrad](https://github.com/Marco-Christiani/zigrad) instead.
- While the core is very simple and self-contained, it is very easy to implement custom operations.

## Acknowledgements

This library is inspired by the following projects:

- [zigrad](https://github.com/Marco-Christiani/zigrad)
- [ZEIN](https://github.com/andrewCodeDev/ZEIN)
- [ggml](https://github.com/ggml-org/ggml)