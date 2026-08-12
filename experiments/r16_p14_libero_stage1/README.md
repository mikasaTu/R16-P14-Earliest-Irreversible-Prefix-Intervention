# R16-P14 LIBERO Stage-1

This directory contains a mechanism-first audit of earliest-irreversible-prefix
intervention on LIBERO.  It deliberately uses a small state-observation
chunked behavior-cloning policy instead of a VLA or a learned risk head.

The development tasks are:

- `open_the_middle_drawer_of_the_cabinet` (mechanism obstruction),
- `put_the_bowl_on_the_plate` (target shift / premature release), and
- `put_the_wine_bottle_on_the_rack` (grasp slip / contact alignment).

`open_the_top_drawer_and_put_the_bowl_inside` is held out as a compositional
task.  The action chunk length is 16; clean evaluation compares execution
horizons 1, 4, 8, and 16.

All large or persistent outputs live under the new CPFS namespaces declared in
`preregistration.yaml`.  `/workspace/leon` is used only for the private PAI
credential file and the user-level LIBERO config during bounded development
smokes.
