#!/bin/bash

for sw in s1 s2 s3; do
  sudo ovs-vsctl clear Bridge $sw flow_tables

  sudo ovs-vsctl \
    -- --id=@ft create Flow_Table \
         name="${sw}_table0" \
         flow_limit=50 \
         overflow_policy=evict \
    -- set Bridge $sw flow_tables:0=@ft

  echo "[$sw] Configurato"
done

echo -e "\n=== Verifica OVSDB ==="
sudo ovs-vsctl list Flow_Table