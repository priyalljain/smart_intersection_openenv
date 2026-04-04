# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from openenv.core.env_server import create_app
from .my_environment import TrafficControlEnv, TrafficAction, TrafficObservation

# Create the FastAPI app
app = create_app(
    env_class=TrafficControlEnv,
    action_model=TrafficAction,
    observation_model=TrafficObservation,
)

def main():
    """Main entry point for openenv serve."""
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()