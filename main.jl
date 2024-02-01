using Plots
using LinearAlgebra

# Parameters    
N = 3 # particle number
ilast = 10000
dt = 0.01
Lx = 10; Ly = 10;
Nx = 21; Ny = 21;
eps_x = Lx/Nx; eps_y = Ly/Ny
anime_bool = true #false

area_xaxis = LinRange(eps_x/2, Lx-eps_x/2, Nx+1); area_yaxis = LinRange(eps_y/2, Ly-eps_y/2, Ny+1)

function initialize()
    global positions, velocities, dt, previous_positions, area, area_tobeplot

    velocities = rand(2, N).-0.5 
    for i in 1:N
        velocities[:, i] = velocities[:, i] ./ norm(velocities[:, i])
    end

    position_generate()
    previous_positions = positions

    area = zeros(Nx+3, Ny+3)
    
    for i in 2:Nx+2
        for j in 2:Ny+2
            dist_list = ones(N) * sqrt(Lx^2 + Ly^2)
            xpos = (i-3/2) * eps_x; ypos = (j-3/2) * eps_y
            for k in 1:N 
                xpos_k = positions[1,k]; ypos_k = positions[2,k]
                dist_list[k] = sqrt((xpos-xpos_k)^2 + (ypos-ypos_k)^2)
            end
            argmin_dist = argmin(dist_list)
            area[i,j] = argmin_dist 
        end
    end

    area_tobeplot = transpose(area[2:Nx+2, 2:Ny+2])

    for i in 1:N
        xpos = positions[1,i]; ypos = positions[2,i]
        a_ind = area_index(xpos, ypos)
        if a_ind != i
           for j in 2:Nx+2
                for k in 2:Ny+2
                    if area[j,k] == i
                        positions[1,i] = (j-3/2) * eps_x
                        positions[2,i] = (k-3/2) * eps_y
                    end
                end
            end
        end
    end
end

function position_generate()
    global positions, Lx, Ly, N
    eps = sqrt(Lx^2 + Ly^2)/N/3
    positions = zeros(2, N)
    for i in 1:N
        theta = rand()*2*pi
        diffx = eps*cos(theta); diffy = eps*sin(theta)
        if i == 1
            posx = Lx/2 + diffx
            posy = Ly/2 + diffy
        else
            posx = positions[1, i-1] + diffx
            posy = positions[2, i-1] + diffy
        end
        if posx < 0
            posx = posx + Lx
        elseif posx > Lx
            posx = posx - Lx
        end
        positions[1, i] = posx
        positions[2, i] = posy
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

function flip_area(x,y,i)
    global area, Nx, Ny, Lx, Ly, area_tobeplot, N, positions
    indx = Int(ceil(x*(Nx+1)/Lx)) + 1
    indy = Int(ceil(y*(Ny+1)/Ly)) + 1
    indx = max(indx, 1); indx = min(indx, Nx+3)
    indy = max(indy, 1); indy = min(indy, Ny+3)

    for j in 1:N
        if j != i
            xpos_j = positions[1,j]; ypos_j = positions[2,j]
            indx_j = Int(ceil(xpos_j*(Nx+1)/Lx)) + 1; indy_j = Int(ceil(ypos_j*(Ny+1)/Ly)) + 1
            if indx_j == indx && indy_j == indy
                return
            end
        end
    end

    area[indx, indy] = i
    area_tobeplot = transpose(area[2:Nx+2, 2:Ny+2])
end

function collision_area()
    global positions, velocities, Nx, Ny, Lx, Ly, area_xaxis, area_yaxis, area

    pos_xs = positions[1, :]; pos_ys = positions[2, :]
    pos_xs_prev = previous_positions[1, :]; pos_ys_prev = previous_positions[2, :]
    tobe_flipped = ones(2, N) * -100 * Lx * Ly 

    for i in 1:N
        pos_x = pos_xs[i]; pos_y = pos_ys[i]
        pos_x_prev = pos_xs_prev[i]; pos_y_prev = pos_ys_prev[i]
        
        a_ind = area_index(pos_x, pos_y)
        a_ind_prev = area_index(pos_x_prev, pos_y_prev)
        # when the particle moves to another area
        if a_ind != a_ind_prev
            if (a_ind_prev != area_index(pos_x,pos_y_prev))
                velocities[1,i] = -velocities[1,i]
            end
            if (a_ind_prev != area_index(pos_x_prev,pos_y))
                velocities[2,i] = -velocities[2,i]
            end
            if a_ind != 0
                #flip_area(pos_x, pos_y, i)
                tobe_flipped[:, i] = [pos_x, pos_y]
            end
            positions = previous_positions .+ velocities * dt
        end
    end
    for i in 1:N
        if tobe_flipped[1, i] > -1 && tobe_flipped[2, i] > -1
            flip_area(tobe_flipped[1, i], tobe_flipped[2, i], i)
        end
    end
end

function compute_scores()
    global area, Nx, Ny, N
    scores = zeros(N)
    for i in 1:N
        scores[i] = sum(area .== i)
    end
    return scores
end

#main loop
i = 0
initialize()
if anime_bool
    anim = Animation()
end
#generate list of colors for each N particles
colors = [RGB(rand(), rand(), rand()) for i in 1:N]

while i<ilast
	global positions, velocities, i, Lx, Ly, Nx, Ny, dt, area_xaxis, area_yaxis, area, N

    update_positions() # update positions
    collision_area() # judge collision with area

    scatter_handle = heatmap(area_xaxis, area_yaxis, area_tobeplot, opacity=0.5, range=(0, N+1), colorbar=true, aspect_ratio=:equal, xlims=(0, Lx), ylims=(0, Ly), legend=false, color=colors)

    scatter!(scatter_handle, positions[1, :], positions[2, :], legend=false, xlims=(0, Lx), ylims=(0, Ly), aspect_ratio=:equal, markersize=10, color=colors)

    if anime_bool
        if i%10 == 0
            frame(anim, scatter_handle)
        end
    else
        display(scatter_handle)
    end
    
    i += 1
end

if anime_bool
    gif(anim, "anime.gif", fps = 1000)
end



