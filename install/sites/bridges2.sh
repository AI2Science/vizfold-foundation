#!/bin/bash

# PSC Bridges-2 ("bridges2"). AF2 mirror in <site>.json; account (= default assoc) is also the /ocean project dir.

slurm::prefix() { PREFIX_DEFAULT=/ocean/projects/$(slurm::default_account)/$USER/openfold; }
