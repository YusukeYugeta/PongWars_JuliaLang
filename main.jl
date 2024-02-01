using Plots
using LinearAlgebra

# Parameters    
N = 50 # group size
M = 2 # number of groups
p = 1.0 # diffusive parameter
ilast = 500000
dt = 0.1
Lx = 15; Ly = 15;
Nx = Lx; Ny = Ly;
eps_x = Lx/Nx; eps_y = Ly/Ny
anime_bool = false#true

area_xaxis = LinRange(eps_x/2, Lx-eps_x/2, Nx+1); area_yaxis = LinRange(eps_y/2, Ly-eps_y/2, Ny+1)

function initialize()
    global positions, velocities, dt, previous_positions, area, area_tobeplot

    velocities = rand(2, N*M).-0.5 
    for i in 1:M
        for j in 1:N
            ind = (i-1)*N+j
            velocities[:, ind] = velocities[:, ind] ./ norm(velocities[:, ind])
        end
    end

    position_generate()
    previous_positions = positions

    area = zeros(Nx+3, Ny+3)

    for i in 2:Nx+2
        for j in 2:Ny+2

            dist_list = ones(M) * sqrt(Lx^2 + Ly^2)
            dist_list_2 = ones(N) * sqrt(Lx^2 + Ly^2)
            xpos = (i-3/2) * eps_x; ypos = (j-3/2) * eps_y
            for k in 1:M
                for l in 1:N
                    xpos_k = positions[1,(k-1)*N+l]; ypos_k = positions[2,(k-1)*N+l]
                    dist_list_2[l] = sqrt((xpos-xpos_k)^2 + (ypos-ypos_k)^2)
                end
                dist_list[k] = minimum(dist_list_2)
            end
            argmin_dist = argmin(dist_list)
            area[i,j] = Int(argmin_dist)
        end
    end

    area_tobeplot = transpose(area[2:Nx+2, 2:Ny+2])

end

function position_generate()
    global positions, Lx, Ly, N
    #eps = sqrt(Lx^2 + Ly^2)/4
    #positions = zeros(2, N)
    #for i in 1:N
    #    theta = rand()*2*pi
    #    diffx = eps*cos(theta); diffy = eps*sin(theta)
    #    posx = mod(diffx, Lx); posy = mod(diffy, Ly)
    #    positions[1, i] = posx
    #    positions[2, i] = posy
    #end
    positions = zeros(2, N*M)
    ind = 1
    R = min(Lx, Ly)/2
    for i in 1:M
        for j in 1:N
            positions[1, ind] = Lx/2 + R/3 * cos(2*pi*i/M)    
            positions[2, ind] = Ly/2 + R/3 * sin(2*pi*i/M)    
            ind += 1
        end 
    end
end

function update_positions()
    global positions, velocities, dt, previous_positions
    previous_positions = positions
    positions = previous_positions .+ velocities * dt
end

function area_index(x,y)
    global area, Nx, Ny, Lx, Ly
    indx = Int(ceil(x*(Nx+1)/Lx)) + 1
    indy = Int(ceil(y*(Ny+1)/Ly)) + 1
    indx = max(indx, 1); indx = min(indx, Nx+3)
    indy = max(indy, 1); indy = min(indy, Ny+3)
    return area[indx, indy]
end

function flip_area(x,y,ind)
    global area, Nx, Ny, Lx, Ly, area_tobeplot, N, positions, velocities
    indx = Int(ceil(x*(Nx+1)/Lx)) + 1
    indy = Int(ceil(y*(Ny+1)/Ly)) + 1
    indx = max(indx, 1); indx = min(indx, Nx+3)
    indy = max(indy, 1); indy = min(indy, Ny+3)

    for i in 1:M
        for j in 1:N
            if i != Int(ind) 
                xpos_j = positions[1,(i-1)*N+j]; ypos_j = positions[2,(i-1)*N+j]
                indx_j = Int(ceil(xpos_j*(Nx+1)/Lx)) + 1; indy_j = Int(ceil(ypos_j*(Ny+1)/Ly)) + 1
                if abs(indx_j - indx)<1 && abs(indy_j - indy)<1
                    return
                end
            end
        end
    end
    
    area[indx, indy] = Int(ind)
    area_tobeplot = transpose(area[2:Nx+2, 2:Ny+2])
end

function collision_area()
    global positions, velocities, Nx, Ny, Lx, Ly, area_xaxis, area_yaxis, area

    pos_xs = positions[1, :]; pos_ys = positions[2, :]
    pos_xs_prev = previous_positions[1, :]; pos_ys_prev = previous_positions[2, :]
    tobe_flipped = ones(3, N*M) * -100 * Lx * Ly 

    ind = 1
    for i in 1:M
        for j in 1:N
            pos_x = pos_xs[ind]; pos_y = pos_ys[ind]
            pos_x_prev = pos_xs_prev[ind]; pos_y_prev = pos_ys_prev[ind]
        
            a_ind = area_index(pos_x, pos_y)
            # when the particle moves to another area
            if a_ind != i
                if (i != area_index(pos_x,pos_y_prev))
                    velocities[1,ind] = -velocities[1,ind]
                end
                if (i != area_index(pos_x_prev,pos_y))
                    velocities[2,ind] = -velocities[2,ind]
                end
                if a_ind != 0
                    #flip_area(pos_x, pos_y, i)
                    tobe_flipped[:, ind] = [pos_x, pos_y, i]
                end
               
                positions = previous_positions .+ velocities * dt
            end
            ind += 1
        end
    end
    for ind in 1:N*M
        if tobe_flipped[1, ind] > -1 && tobe_flipped[2, ind] > -1
            flip_area(tobe_flipped[1, ind], tobe_flipped[2, ind], tobe_flipped[3, ind])
        end
    end
end

function get_scores()
    global positions, velocities, Nx, Ny, Lx, Ly, area_xaxis, area_yaxis, area
    scores = zeros(M)
    for i in 1:Nx
        for j in 1:Ny
            scores[Int(area[i+1, j+1])] += 1/Nx/Ny
        end
    end
    return scores
end

function check_consistency()
    global positions, velocities, Nx, Ny, Lx, Ly, area_xaxis, area_yaxis, area
    count = 0
    for i in 1:M
        for j in 1:N
            pos_x = positions[1, (i-1)*N+j]; pos_y = positions[2, (i-1)*N+j]
            a_ind = area_index(pos_x, pos_y)    
            if a_ind != i
                count += 1
            end
        end
    end
    print("Error count: ", count, "\n")
end

#main loop
i = 0
initialize()
if anime_bool
    anim = Animation()
end
#generate list of colors for each N particles
#colors = [RGB(rand(), rand(), rand()) for j in 1:N for i in 1:M]
colors = [RGB(1, 0, 0) for j in 1:N*M]
colors_heatmap = [RGB(1,0,0) for j in 1:M]
for i in 1:M
    color = RGB(rand(), rand(), rand())
    colors_heatmap[i] = color
    for j in 1:N
        colors[(i-1)*N+j] = color
    end
end


for i in 1:ilast
	global positions, velocities, Lx, Ly, Nx, Ny, dt, area_xaxis, area_yaxis, area, N

    update_positions() # update positions
    collision_area() # judge collision with area

    scatter_handle = heatmap(area_xaxis, area_yaxis, area_tobeplot, opacity=0.5, range=(0, N+1), colorbar=true, aspect_ratio=:equal, xlims=(0, Lx), ylims=(0, Ly), legend=false, color=colors_heatmap)

    scatter!(scatter_handle, positions[1, :], positions[2, :], legend=false, xlims=(0, Lx), ylims=(0, Ly), aspect_ratio=:equal, markersize=5, color=colors)

    if i%100 == 1
        println("step: ", i)
        println("scores: ", get_scores())
        check_consistency()
    end

    if anime_bool
        if i%20 == 1
            frame(anim, scatter_handle)
        end
    else
        display(scatter_handle)
    end
    
end

if anime_bool
    gif(anim, "anime.gif", fps = 60)
end



