#!/bin/bash

# PSC Bridges-2. AF2 mirror in <site>.json; the account (= default assoc) is also the /ocean project dir the prefix templates off.

slurm::discover() { export OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$(slurm::default_account)}; }
