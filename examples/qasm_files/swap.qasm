// SWAP gate impemented in terms of cx's

OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];

cx q[1], q[0];
cx q[0], q[1];
cx q[1], q[0];
