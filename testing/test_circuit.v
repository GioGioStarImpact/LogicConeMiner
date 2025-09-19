// Simple test circuit for LogicConeMiner
module test_circuit (
    input clk,
    input a, b, c, d,
    output reg q1, q2,
    output y1, y2, y3
);

// Sequential elements
always @(posedge clk) begin
    q1 <= a & b;
    q2 <= c | d;
end

// Combinational logic forming various cones
wire w1, w2, w3, w4;

// Basic gates
AND2 u1 (.A(a), .B(b), .Y(w1));
OR2 u2 (.A(c), .B(d), .Y(w2));
XOR2 u3 (.A(w1), .B(w2), .Y(w3));
NOT u4 (.A(w3), .Y(w4));

// More complex logic
AND2 u5 (.A(w1), .B(w4), .Y(y1));
OR3 u6 (.A(w2), .B(w3), .C(w4), .Y(y2));
XOR2 u7 (.A(a), .B(w4), .Y(y3));

endmodule

// Library cells
module AND2(input A, B, output Y);
    assign Y = A & B;
endmodule

module OR2(input A, B, output Y);
    assign Y = A | B;
endmodule

module OR3(input A, B, C, output Y);
    assign Y = A | B | C;
endmodule

module XOR2(input A, B, output Y);
    assign Y = A ^ B;
endmodule

module NOT(input A, output Y);
    assign Y = ~A;
endmodule