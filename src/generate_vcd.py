from myhdl import block, delay, instance, Signal, intbv, always_comb

@block
def qst_hardware_logic(clk, x_in, y_in, z_in, ready):
    """
    A simplified hardware logic module representing the input 
    stage of our Quantum State Tomography MLP.
    """
    @always_comb
    def logic():
        # Simulating that when all inputs are present, the 'ready' signal flips
        if x_in > 0 and y_in > 0 and z_in > 0:
            ready.next = 1
        else:
            ready.next = 0
            
    return logic

# Simulation Setup
clk = Signal(bool(0))
x = Signal(intbv(0)[8:])
y = Signal(intbv(0)[8:])
z = Signal(intbv(0)[8:])
ready = Signal(bool(0))

inst = qst_hardware_logic(clk, x, y, z, ready)
inst.config_sim(backend='icarus', directory='outputs')

@block
def testbench():
    instance_to_test = qst_hardware_logic(clk, x, y, z, ready)
    
    @instance
    def stimulus():
        # Simulating the change of measurement signals over time
        for i in range(5):
            x.next = 10 * i
            y.next = 20 * i
            z.next = 30 * i
            yield delay(10)
            print(f"Time {i*10}: X={x}, Y={y}, Z={z}, Ready={ready}")

    return instance_to_test, stimulus

# Running the simulation and dumping to .vcd
tb = testbench()
tb.config_sim(trace=True, directory='outputs')
tb.run_sim(50)
print("Simulation complete. outputs/testbench.vcd generated.")