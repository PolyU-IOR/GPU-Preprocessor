* HPR integration fixture: unique x = [0.25, 1.0, 1.75, 0.75, 1.5]
* Objective offset 7.25; active lower/upper rows and variable bounds.
NAME HPRPIPE
ROWS
 N OBJ
 G R0
 L R1
 E R2
 L R3
COLUMNS
    X0 OBJ -4
    X0 R0 -0.5
    X0 R1 2
    X0 R2 1
    X0 R3 -2.5
    X1 OBJ -5.90625
    X1 R0 1.5
    X1 R1 0.5
    X1 R2 -2
    X1 R3 1
    X2 OBJ -8.3125
    X2 R0 2.5
    X2 R1 -1.5
    X2 R2 0.5
    X2 R3 2
    X3 OBJ -9.0625
    X3 R0 -1
    X3 R1 2.5
    X3 R2 1.5
    X3 R3 -0.5
    X4 OBJ -13.59375
    X4 R0 2
    X4 R1 1
    X4 R2 -2.5
    X4 R3 1.5
RHS
    RHS1 OBJ -7.25
    RHS1 R0 8
    RHS1 R1 1.75
    RHS1 R2 -3.5
    RHS1 R3 7.75
RANGES
    RNG1 R0 2
    RNG1 R1 1
    RNG1 R3 3
BOUNDS
 FX BND X0 0.25
 LO BND X1 1.0
 UP BND X1 4.0
 LO BND X2 -0.25
 UP BND X2 1.75
 LO BND X3 -1.25
 UP BND X3 3.75
 LO BND X4 -0.5
 UP BND X4 4.5
QUADOBJ
    X0 X0 4
    X0 X1 0.25
    X0 X2 0.375
    X0 X3 0.125
    X0 X4 0.25
    X1 X1 5
    X1 X2 0.125
    X1 X3 0.25
    X1 X4 0.375
    X2 X2 6
    X2 X3 0.375
    X2 X4 0.125
    X3 X3 7
    X3 X4 0.25
    X4 X4 8
ENDATA
