// Test circuit with macro instances
module test_with_macro (
    input clk, rst, a, b, c, d,
    output y1, y2, y3
);

// Standard cell instances (defined in CSV)
wire w1, w2, w3, w4, w5;

AND2 u1 (.A(a), .B(b), .Y(w1));
OR2 u2 (.A(c), .B(d), .Y(w2));
NAND2 u3 (.A(w1), .B(w2), .Y(w3));

// Macro instance (NOT in CSV - should be treated as boundary)
CPU_CORE cpu_inst (
    .clk(clk),
    .reset(rst),
    .data_in(w3),
    .enable(1'b1),
    .data_out(w4),
    .status(w5)
);

// More standard cells after macro
XOR2 u4 (.A(w4), .B(a), .Y(y1));
OR3 u5 (.A(w5), .B(w1), .C(w2), .Y(y2));

// Another macro
MEMORY_CTRL mem_ctrl (
    .clk(clk),
    .addr(w4),
    .data_out(y3)
);

endmodule