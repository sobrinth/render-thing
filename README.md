# Vulkan Renderer

Where I try to write a renderer in Rust using [Ash][0].

Starting off with getting a Vulkan renderer going and looking at how to adapt that afterward.

Let's see where this will lead...


## Run it
To be able to run you need to have the Vulkan SDK installed or another way to compile the shaders located in `assets/shaders`

The targets currently try to load the 'Sponza' scene from the `assets/models/downloaded/sponza/` directory.
The application will start without them present but just shows nothing :)

To change you'd have the calls to `scene::load_gltf()` in the respective main.rs files of the crates

You can download them (and more) from this [Repo][3]

When "running" use F1-F4 to bring up some debug windows and scene information


There are currently two different crates you can run:
#### 'game' crate
Simple application with a simple moving player and some moving meshes

```cargo run --release -p game```
#### 'application' crate
Just a simple application with a flying camera

``` cargo run --release```
## Credits
- The initial Vulkan 1.0 version is based on the good old [Vulkan Tutorial][1] by Alexander Overvoorde and specifically [this Rust implementation][2] by Adrien Ben
- "Newer" version with dynamic rendering based on [VkGuide][4]

[0]: https://github.com/MaikKlein/ash
[1]: https://vulkan-tutorial.com/Introduction
[2]: https://github.com/adrien-ben/vulkan-tutorial-rs
[3]: https://github.com/KhronosGroup/glTF-Sample-Assets
[4]: https://vkguide.dev/