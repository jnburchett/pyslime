pyslime
=======

This software package enables simple loading and analysis of output from Monte Carlo Physarum Machine models.

For basic usage:
```
from pyslime import slime
slimeobj = slime.Slime.from_dir(mydir)
```
where `mydir` is the path to the full MCPM output directory