<div align="center">
    <h1><code>brainz</code></h1>
    <i>Braaaaaaaaaaaaaiiiiiiiinnnnnnnnsssss 🧠🧟‍♂️</i>
    <br/>
    <i>A small tensor library with just enough operators for Deep Learning written in pure Zig</i>
    <hr>
</div>

> **Warning**
> This library is a work in progress. Design considerations aren't set in stone.


This library implements a `Tensor` primitive and a subset of operators to design, train and run deep learning models using Zig, on any platform that runs Zig code.

See [ops.zig](src/ops.zig) for a list of currently implemented operators. Also note that making your own is quite straightforward too!


The library provides an abstraction for multithreading / accelerator support with _explicit synchronization_ through the `Device` interface. Operators take a device as parameter and are dispatched in a non-blocking way. 
**Synchronization is totally explicit and it is up to the programmer to between insert sync points between dependent computations**.

A multi-threaded CPU device is currently available, but we could maybe get some juicy GPU accel someday to go zoomin (soon-ish™).

## Installation

After adding the project to your `build.zig.zon`, add to your `build.zig`

```
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOptions(.{});

    const brainz = b.dependency("brainz", .{
        .optimize = optimize,
        .target = target,
    });

    your_project_executable.root_module.addImport("brainz", brainz.module("brainz"));
}

```


## Getting started
 Please check the `examples/` directory to get started.

 Want something a little bigger to look at ? See [MNIST from scratch](https://github.com/Game4all/mnist-from-scratch)
