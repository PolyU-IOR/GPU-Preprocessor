NAME          TINYQP
ROWS
 N  OBJ
 E  BALANCE
COLUMNS
    X1        OBJ       -4
    X2        OBJ       -6       BALANCE   1
    X3        OBJ       -2       BALANCE   1
RHS
    RHS1      BALANCE   4
BOUNDS
 FX BND1      X1        1
 LO BND1      X2        0
 LO BND1      X3        0
QUADOBJ
    X1        X1        2
    X2        X2        2
    X3        X3        2
ENDATA
