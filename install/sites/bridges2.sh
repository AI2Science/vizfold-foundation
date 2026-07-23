#!/bin/bash

# PSC Bridges-2 ("bridges2"). AF2 mirror in <site>.json; account (= default assoc) is also the /ocean project dir.

site::prefix() {
    local a=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}
    PREFIX_DEFAULT=/ocean/projects/$a/$USER/openfold
}
