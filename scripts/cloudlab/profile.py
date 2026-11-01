"""rtier on CloudLab: a control node and the data nodes' hosts on one experiment LAN.

node0 is the control machine (10.10.1.1): it builds the code and runs scripts/run_cluster.py,
the controller and the load generator. node1 to nodeN are the hosts of the data nodes n1 to nN
(10.10.1.2 to 10.10.1.N+1), the order of an experiment's Hosts.Nodes. The addresses are fixed
so that the experiment files can name the hosts before the nodes exist
(configs/experiment.*-cloudlab-*.json).

Instructions:
On every node, run scripts/cloudlab/setup_node.sh (node0 with --control). Then, from a machine
that logs in to all of them, run scripts/cloudlab/authorize_control.sh. The steps are on the
wiki page Running on CloudLab.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
pc.defineParameter("hosts", "Hosts of data nodes (node1 to nodeN; node0, the control node, comes on top)",
                   portal.ParameterType.INTEGER, 8)
pc.defineParameter("hwtype", "Hardware type of every node: one type, so that the control node's "
                   "-march=native build runs on every host", portal.ParameterType.STRING, "c220g5")
pc.defineParameter("image", "Disk image URN (empty: the hardware type's default image)",
                   portal.ParameterType.STRING, "")
params = pc.bindParameters()
if not 1 <= params.hosts <= 200:
    pc.reportError(portal.ParameterError("between 1 and 200 hosts", ["hosts"]))
pc.verifyParameters()

request = pc.makeRequestRSpec()
lan = request.LAN("lan")
for i in range(params.hosts + 1):
    node = request.RawPC("node%d" % i)
    node.hardware_type = params.hwtype
    if params.image:
        node.disk_image = params.image
    iface = node.addInterface("if1")
    iface.addAddress(pg.IPv4Address("10.10.1.%d" % (i + 1), "255.255.255.0"))
    lan.addInterface(iface)

pc.printRequestRSpec(request)
