#!/bin/bash

sudo ovs-vsctl -- --id=@ft create Flow_Table flow_limit=50 overflow_policy=refuse -- set Bridge s1 flow_tables:0=@ft &> /dev/null
sudo ovs-vsctl -- --id=@ft create Flow_Table flow_limit=50 overflow_policy=refuse -- set Bridge s2 flow_tables:0=@ft &> /dev/null
sudo ovs-vsctl -- --id=@ft create Flow_Table flow_limit=50 overflow_policy=refuse -- set Bridge s3 flow_tables:0=@ft &> /dev/null

echo -e "=== Flow Table Sizes ==="

sudo ovs-vsctl list Flow_Table | grep -E "^(_uuid|flow_limit|overflow_policy)"

