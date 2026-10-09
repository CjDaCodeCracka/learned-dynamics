

function euler(f, x0, time_start, time_end ,dt) #IC, time interval

    @assert isfinite(dt) && dt > 0 "dt must be finite and positive"
    @assert isfinite(time_start) && isfinite(time_end) "Times must be finite"
    @assert time_end != time_start "Start and end times must differ"
    
    direction = sign(time_end - time_start) #tells us to go fwd or bwd


    times = collect(time_start:direction * dt:time_end) #time interval, from start to end with step size dt
    
    #Adds a shorter final time step if we need one
    if !isapprox(times[end], time_end)
        push!(times, time_end)
    else
        times[end] = time_end
    end

    
    states = zeros(length(times))# saves our x at each point
    derivatives = zeros(length(times))# saves our dx/dt at each point
    states[1] = x0 #intial condition

    for i in 1:length(times) 
        derivatives[i] = f(states[i]) 
    #at the current step: 
    #use our dx/dt function at current x for dx/dt at that point
        if i < length(times) #checks if we're still in the time interval
            h = times[i+1] - times[i] #Takes care of shorter step at the end
            states[i+1] = states[i] + h * derivatives[i] #euler's method for next x point
        end
    end

    return (; times, states, derivatives) 
    #; creates a list of our simulation data with times, states, and derivatives 
end