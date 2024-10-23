import jax
from jax import numpy as jp
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

import jax

from brax.io import mjcf

ball = mjcf.loads(
    """<mujoco>
         <option timestep="0.005"/>
         <worldbody>
           <body pos="0 0 3">
             <joint type="free"/>
             <geom size="0.5" type="sphere"/>
           </body>
           <geom size="40 40 40" type="plane"/>
         </worldbody>
       </mujoco>
  """)
