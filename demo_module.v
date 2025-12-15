module simple_counter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    output reg  [3:0]  count
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 4'b0000;
    else if (enable)
        count <= count + 1'b1;
    // else retain current value
end

endmodule
