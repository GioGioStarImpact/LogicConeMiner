// Multi-module test case for LogicConeMiner
// Demonstrates new features: multiple modules, complex port declarations, macros

// CPU 核心模組（作為 Macro 處理）
module CPU_CORE(
    input clk, reset,
    input [31:0] instruction,
    output [31:0] pc,
    output reg_write
);
    // 內部只有不在 cell_library 中的複雜元件
    COMPLEX_ALU alu_inst (.a(pc), .b(instruction), .out(reg_write));
    // 這會被識別為 Macro 模組
endmodule

// 記憶體控制模組（作為 Macro 處理）
module MEMORY_CTRL(
    input clk,
    input [15:0] addr,
    output [31:0] data
);
    SRAM_CONTROLLER ctrl (.clk(clk), .addr(addr), .data(data));
endmodule

// 主處理器模組（標準模組，含標準邏輯閘）
module PROCESSOR(
    input clk, reset,
    input a, b, c,
    output d, e
);
    // 複雜的埠宣告
    input enable, mode;
    output status, ready;
    wire internal_sig, temp_wire;
    wire bus_signal[15:0];

    // 標準邏輯閘（會在 cell_library.csv 中定義）
    AND2 u1 (.A(a), .B(b), .Y(internal_sig));
    OR2 u2 (.A(internal_sig), .B(c), .Y(temp_wire));
    NAND2 u3 (.A(temp_wire), .B(enable), .Y(status));

    // DFF 時序元件
    DFF ff1 (.D(status), .CLK(clk), .Q(ready));

    // 使用 CPU_CORE 作為 Macro
    CPU_CORE cpu_inst (
        .clk(clk),
        .reset(reset),
        .instruction({16'b0, bus_signal}),
        .pc(d),
        .reg_write(e)
    );

    // 更多組合邏輯
    XOR2 u4 (.A(ready), .B(mode), .Y(bus_signal[0]));
    NOT u5 (.A(bus_signal[0]), .Y(bus_signal[1]));

endmodule

// 頂層模組
module TOP_MODULE(
    input sys_clk, sys_reset,
    input [3:0] input_data,
    output [3:0] output_data,
    output system_ready
);

    wire proc_d, proc_e, proc_ready;
    wire [31:0] mem_data;

    // 實例化處理器
    PROCESSOR proc (
        .clk(sys_clk),
        .reset(sys_reset),
        .a(input_data[0]),
        .b(input_data[1]),
        .c(input_data[2]),
        .enable(input_data[3]),
        .mode(1'b1),
        .d(proc_d),
        .e(proc_e),
        .status(output_data[0]),
        .ready(proc_ready)
    );

    // 實例化記憶體控制器（Macro）
    MEMORY_CTRL mem_ctrl (
        .clk(sys_clk),
        .addr({12'b0, input_data}),
        .data(mem_data)
    );

    // 簡單的輸出邏輯
    AND2 out_logic1 (.A(proc_d), .B(mem_data[0]), .Y(output_data[1]));
    OR2 out_logic2 (.A(proc_e), .B(mem_data[1]), .Y(output_data[2]));
    BUF out_buf (.A(proc_ready), .Y(system_ready));

    // 確保 output_data[3] 有驅動
    NOT final_inv (.A(system_ready), .Y(output_data[3]));

endmodule