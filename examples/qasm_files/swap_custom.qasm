// SWAP gate defined as a custom gate.

OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

gate swap a, b{
cx b, a;
cx a, b;
cx b, a;
}

swap q[0], q[1]

measure q -> c
