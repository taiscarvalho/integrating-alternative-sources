#Instalação dos pacotes
#Pkg.clone("https://github.com/odow/SDDP.jl.git")

#Pkg.update()                                      
#workspace()                                       

using SDDP, GLPK, Statistics, Plots, DelimitedFiles, DataFrames, GLM, XLSX, CSV, StatsBase, PlotThemes, Interpolations, Random, StatsPlots
#using Dierckx
using LaTeXStrings
using Plots.PlotMeasures

"Data"

nA = 30                            #número de anos no horizonte de planejamento
nM = nA*12                         #número de meses
months = repeat(collect(1:12), nA) #vetor com meses p/ iterar

n_res = 9                          #número de reservatórios
n_altern = 3                       #número de fontes alternativas
nF = n_altern + n_res              #número de fontes hídricas

n_sims = 100                       #número de séries de vazão simuladas
alfa = 0.8                         #porcentagem de água consumida que vira esgoto
conv = (30*24*3600)/10^6           #converte de m³/seg para hm³/mês

cap_reuso_max = 4.5*conv     #hm³/mês
cap_dessal_max = 1*conv      #hm³/mês
cap_reusocinz_max = 0.3*conv #hm³/mês 
cap_pisf_max = 7*conv        #hm³/mês
initial_cap = [0, 0, 0]      #hm³/mês

cap_max = [cap_reuso_max, cap_dessal_max, cap_pisf_max]

"Costs"

# Failure cost

beta = 8 #RMF
beta_pecem = 8
beta_irr = 6
pen = 10

aom_var_cost = [2.47-1.64, 1.8, 0.51, 0]/conv      #R$/hm³/mês
aom_fix_cost = [0.6, 1.01, 0, 0.69]/conv      #R$/hm³/mês
inst_cost = [1.83, 1.12, 10.2, 0]/conv           #R$/hm³/mês

aom_var_cost = [2.47-1.64, 1.8, 0.51]/conv      #R$/hm³/mês
aom_fix_cost = [0.6, 1.01, 0.69]/conv      #R$/hm³/mês
inst_cost = [1.83, 1.12, 0]/conv           #R$/hm³/mês
transf_cost = fill(0.03*conv, n_res) #R$/hm³/mês

"Water demands"

demand_hum = Array{Any}(undef, n_res)
demand_hum[1] = LinRange(0.07, 0.09, nM)*conv   #Banabuiú
demand_hum[2] = LinRange(0, 0, nM)*conv         #Aracoiaba
demand_hum[3] = LinRange(1.2, 3, nM)*conv       #Sítios Novos

demand_hum[4] = LinRange(0.23, 0.28, nM)*conv            #Orós
demand_hum[5] = LinRange(0.61+0.13, 0.73+0.15, nM)*conv  #Castanhão + Canal do Trabalhador
demand_hum[6] = LinRange(0, 0, nM)*conv                  #Pacajus
demand_hum[7] = LinRange(0, 0, nM)*conv                  #Pacoti
demand_hum[8] = LinRange(0, 0, nM)*conv                  #Riachão
demand_hum[9] = LinRange(9+0.27, 18+0.33, nM)*conv  #Gavião + Eixão das águas


demand_irr = Array{Any}(undef, n_res)
demand_irr[1] = LinRange(0.92, 1.04, nM)*conv   #Banabuiú #1.62
demand_irr[2] = LinRange(0, 0, nM)*conv         #Aracoiaba
demand_irr[3] = LinRange(0, 0, nM)*conv         #Sítios Novos

demand_irr[4] = LinRange(3.43, 3.73, nM)*conv   #Orós #5.82
demand_irr[5] = LinRange(11.93, 13.87, nM)*conv #Castanhão
demand_irr[6] = LinRange(0, 0, nM)*conv         #Pacajus
demand_irr[7] = LinRange(0, 0, nM)*conv         #Pacoti
demand_irr[8] = LinRange(0, 0, nM)*conv         #Riachão
demand_irr[9] = LinRange(0, 0, nM)*conv         #Gavião

"Parameters of surface reservoirs"

# Inflow (m³/s)

reservatorios = ["Banabuiu", "Aracoiaba", "Sitios", "Oros", "Castanhao", "Pacajus", "Pacoti", "Riachao", "Gaviao"]

tst = Array{Any}(undef, n_res)
for i in 1:n_res
    local inflow = DataFrame(XLSX.readtable("data/inflow_series.xlsx", reservatorios[i])) .* conv
    serie = inflow[361:720, 1:100]
    tst[i] = [serie[:,c] for c in 1:size(serie,2)]
end

probs_inflow = repeat([1/n_sims], n_sims)

# Connection between reservoirs

b = [0.0, 1, 0.0, 1, 1, 1, 1, 1, 0.0] #índice que indica se o reservatório transfere águas para outro
c = [0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1] #índice que indica se o reservatório recebe águas de outro

# Volume máximo (10e3 hm³)

#reservatorios = ["Banabuiú", "Aracoiaba", "Sitios Novos", "Orós", "Castanhão", "Pacajus", "Pacoti", "Riachão", "Gavião"]
reservatorios = DataFrame(reservatorio = ["Banabuiú", "Aracoiaba", "Sitios Novos", "Orós", "Castanhão", "Pacajus", "Pacoti", "Riachão", "Gavião"])
volumes_max = DataFrame(CSV.File("data/volume_max_reservoirs.csv"))
volumes_max = leftjoin!(reservatorios, volumes_max, on = :reservatorio)

vol_max_res = volumes_max.volume

initial_vol_res = vol_max_res*0.3

# Evaporation (mm)

evaps = DataFrame(CSV.File("data/evaporation_reservoirs.csv"))
evaps = leftjoin!(reservatorios, evaps, on = :reservatorio)[:,3:14] ./ 10e5

# CAV
# volume = 10e3 hm³ | area = km² -> hm²
reservatorios = DataFrame(reservatorio = ["Banabuiú", "Aracoiaba", "Sitios Novos", "Orós", "Castanhão", "Pacajus", "Pacoti", "Riachão", "Gavião"])
CAVS = DataFrame(CSV.File("data/cav_reservoirs.csv"))
CAVS = innerjoin(reservatorios, select(CAVS, Not("cod")), on = :reservatorio)

CAVS.A = CAVS.A .* 10e2

# Coefficients CAV

coeficientes = Array{Any}(undef, n_res)

function baseLinear(xs,i,x)
    if i == 1
        if xs[1] <= x <= xs[2]
            return (x - xs[2])/(xs[1] - xs[2])
        else
            return 0
        end
    elseif xs[i-1] <= x <= xs[i]
        return (x-xs[i-1])/(xs[i] - xs[i-1])
    elseif xs[i] <= x <= xs[i+1]
        return (xs[i+1] - x)/(xs[i+1] - xs[i])
    elseif i == length(xs)
        if xs[end-1] <= x <= xs[end]
            return (x - xs[end-1])/(xs[end] - xs[end-1])
        else
            return 0
        end
    else
        return 0
    end
end

function interpolLinear(xs, ys)
    n = length(xs)
    function f_interpol(x)
        Interpolation = 0
        for i in 1:n
            Interpolation = Interpolation + (ys[i]*baseLinear(xs,i,x))
        end
        return Interpolation
    end
    return f_interpol
end

interp = Array{Any}(undef, n_res)
cavs = Array{Any}(undef, n_res)
coeficientes = Array{Any}(undef, n_res)
#n_points = 10
coefs = Array{Any}(undef, n_res)
#coefsq = Array{Any}(undef, n_res)

for i = 1:n_res
    local reserv = ["Banabuiú", "Aracoiaba", "Sitios Novos", "Orós", "Castanhão", "Pacajus", "Pacoti", "Riachão", "Gavião"]
    cavs[i] = filter(r -> any(occursin.(reserv[i], r.reservatorio)), CAVS)[:,2:3]
    #interp_linear[i] = interpolLinear(cav.V, cav.A)
    coeficientes[i] = lm(@formula(A ~ V), cavs[i])
    coefs[i] = Array{Any}(undef, 2)
    coefs[i][1] = coef(coeficientes[i])[1]
    coefs[i][2] = coef(coeficientes[i])[2]
end

function Evp_nl(volume, coeficientes, evap, mes, res)
    E = evaps[res,mes]*coeficientes[res][1] +
       evaps[res,mes]*coeficientes[res][2]*volume +
       evaps[res,mes]*coeficientes[res][3]*(volume^2)
    return E
end

# Function to estimate Evapotranspiration
function Evp(volume, coeficientes, evap, mes, res)
    E = evap[res,mes]*coeficientes[res][1] +
       evap[res,mes]*coeficientes[res][2]*volume
    return E
end

"Optimization model"

function expansionmodel(;
    #annualizedcosts::Bool = true
)
    tx = 0.08
    rho = (1 + tx)^(1/12) - 1

    model = SDDP.LinearPolicyGraph(
            stages = nM,
            sense = :Min,
            lower_bound = 0,
            optimizer = GLPK.Optimizer,
        ) do subproblem, stage

    # variáveis de estado
    # capacidade das fontes hídricas
    @variable(
    subproblem,
    0 <= cap[i = 1:n_altern] <= cap_max[i],
    SDDP.State,
    initial_value = initial_cap[i]
    )

    # volume armazenado nos reservatórios superficiais
    @variable(
    subproblem,
    0 <= vol_res[i = 1:n_res] <= vol_max_res[i],
    SDDP.State,
    initial_value = initial_vol_res[i]
    )

    # variáveis de decisão
    @variables(subproblem,
    begin
        x[i = 1:n_altern] >= 0
        y[i = 1:n_altern] >= 0
        cap_inst[i = 1:n_altern] >= 0
        x_hum[i = 1:n_res] >= 0
        x_irr[i = 1:n_res] >= 0
        x_gav_pecem >= 0
        spill[i = 1:n_res] >= 0 #retirada não programada*
        area[i = 1:n_res] >= 0
        deplete[i = 1:n_res] >= 0
        outflow[i = 0:n_res] >= 0
        inflow[i = 1:n_res]
        unmet[i = 1:4] >= 0
        unmet_irr[i = 1:3] >= 0
        unmet_pecem >= 0
        evap_otm[i = 1:n_res]
    end)

    # restrições
    @constraints(subproblem,
    begin
        # a capacidade não pode ser reduzida
        [i in 1:n_altern], cap[i].out >= cap[i].in
        [i in 1:n_altern], cap_inst[i] == cap[i].in + y[i]

        # y representa o incremento na capacidade de expansão
        [i in 1:n_altern], cap[i].out == cap[i].in + y[i]

        # a retirada não pode ser maior que a capacidade da fonte
        [i in 1:n_altern], x[i] <= cap[i].in

        # caso os res. atinjam volume menor que 20%, uma penalidade é cobrada
        [i in 1:n_res], deplete[i] >= (0.2 * vol_max_res[i]) - vol_res[i].out

        # # Retirada não pode ser maior que o volume disponível
        [i in 1:8], x_hum[i] + x_irr[i] <= vol_res[i].in
        x_hum[9] + x_irr[9] + x_gav_pecem <= vol_res[9].in

        # Penalidade pelo não atendimento à demanda da RMF e Pecém

        demand_constraint, unmet[1] >= demand_hum[9][stage] - x[2] - x_hum[9] #RMF, Gavião
        unmet_pecem >= demand_hum[3][stage] - x[1] - x_hum[3] - x_gav_pecem #Pecém, Sítios Novos

        unmet[2] >= demand_hum[1][stage] - x_hum[1] #Banabuiú
        unmet[3] >= demand_hum[4][stage] - x_hum[4] #Orós
        unmet[4] >= demand_hum[5][stage] - x_hum[5] #Castanhão

        unmet_irr[1] >= demand_irr[1][stage] - x_irr[1] #Banabuiú
        unmet_irr[2] >= demand_irr[4][stage] - x_irr[4] #Orós
        unmet_irr[3] >= demand_irr[5][stage] - x_irr[5] #Castanhão

        # A retirada deve ser igual ou menor que a demanda
        x[2] + x_hum[9] <= demand_hum[9][stage] #RMF
        x[1] + x_hum[3] + x_gav_pecem <= demand_hum[3][stage] #Pecém

        # Demanda humana
        x_hum[1] <= demand_hum[1][stage] #Banabuiú
        x_hum[2] == 0 #Aracoiaba
        x_hum[4] <= demand_hum[4][stage] #Orós
        x_hum[5] <= demand_hum[5][stage] #Castanhão
        [i in 6:8], x_hum[i] == 0

        # Demanda agrícola
        x_irr[1] <= demand_irr[1][stage] #Banabuiú
        [i in 2:3], x_irr[i] == 0
        x_irr[4] <= demand_irr[4][stage] #Orós
        x_irr[5] <= demand_irr[5][stage] #Castanhão
        [i in 6:9], x_irr[i] == 0

        x[1] <= (x[2] + x_hum[9]) * alfa # reuso

        end)

        @constraints(subproblem,
        begin
            [i in 1:n_res],
            vol_res[i].out ==
            vol_res[i].in + inflow[i] - x_hum[i] - x_irr[i] -
            Evp(vol_res[i].out, coefs, evaps, months[stage], i) -
            #Evp_nl(vol_res[i].out, coefsq, evaps, months[stage], i) -
            #area[i]*evaps[i,:][months[stage]] -
            spill[i] -
            b[i]*outflow[i] +
            ifelse(i != 1, c[i]*outflow[i-1], 0) +
            ifelse(i == 6, outflow[2], 0) -
            ifelse(i == 9, x_gav_pecem, 0) +
            ifelse(i == 5, x[3], 0)
        end)

        # Restrições: início da operação do reuso e da dessalinização
        if stage in collect(1:1:3*12)
            @constraints(subproblem,
            begin
                y[1] == 0
                y[2] == 0
            end)
        end

        if stage in collect(3*12+1:1:5*12)
            @constraints(subproblem,
            begin
                y[1] == 0
            end)
        end

        @stageobjective(subproblem,
        sum(aom_fix_cost[i] * cap_inst[i] for i = 1:n_altern) +
        sum(aom_var_cost[i] * x[i] for i = 1:n_altern) +
        sum(inst_cost[i] * cap_inst[i] for i = 1:n_altern) +
        sum(transf_cost[i] * outflow[i] for i = 1:n_res) +
        beta * sum(unmet[i] for i = 1:4) +
        beta_irr * sum(unmet_irr[i] for i = 1:3) +
        beta_pecem * unmet_pecem +
        pen * sum(deplete[i] for i = 1:n_res) #+
        #p * sum(variavel[i] for i = 1:n_res)
        )

        SDDP.parameterize(subproblem, 1:n_sims, probs_inflow) do ω
            for i in 1:n_res
                JuMP.fix(inflow[i], tst[i][ω][stage])
            end
        end
    end
end

## Training and simulation

Random.seed!(42)
model = expansionmodel()
SDDP.numerical_stability_report(model)

SDDP.train(model, iteration_limit = 100)

#cvx_comb_measure = 0.5 * SDDP.Expectation() + 0.5 * SDDP.WorstCase()
SDDP.train(model, risk_measure = SDDP.WorstCase(), iteration_limit = 20)
#stopping_rules = [SDDP.BoundStalling(10, 1e10)])

sims = SDDP.simulate(model, 100, [:evap_otm, :cap, :vol_res, :x, :y, :spill, :outflow, :cap_inst, :x_irr, :x_hum, :x_gav_pecem])

#If you simulate the policy, the simulated value is the risk-neutral value of the policy.

# Exportar resultados da simulacao

volume_res =  zeros(0)
evap_otm =  zeros(0)
retirada_hum =  zeros(0)
retirada_irr =  zeros(0)

for k in 1:9
    for i in 1:n_sims
        for j in 1:nM
            append!(evap_otm, sims[i][j][:evap_otm][k])
            append!(volume_res, sims[i][j][:vol_res][k].in)
            if(k==9)
                append!(retirada_hum, sims[i][j][:x_hum][k]+sims[i][j][:x_gav_pecem])
            end
            if(k!=9)
                append!(retirada_hum, sims[i][j][:x_hum][k])
            end
            append!(retirada_irr, sims[i][j][:x_irr][k])
        end
    end
end

df = DataFrame()
df.reservatorio = repeat(reservatorios.reservatorio, inner=n_sims*nM)
df.nomes = repeat(repeat(1:n_sims, inner=nM), outer=n_res)
df.ano = repeat(repeat(repeat(1:nA, inner=12), outer=n_sims), outer=n_res)
df.mes = repeat(repeat(repeat(1:12, outer=nA), outer=n_sims), outer=n_res)
df.retirada_hum = retirada_hum
df.retirada_irr = retirada_irr
df.volume_res = volume_res

CSV.write("results/df_reservatorios.csv", df)

retirada_reuso = zeros(0)
retirada_dessal = zeros(0)
retirada_pisf = zeros(0)

for i in 1:n_sims
    for j in 1:nM
        append!(retirada_reuso, sims[i][j][:x][1])
        append!(retirada_dessal, sims[i][j][:x][2])
        append!(retirada_pisf, sims[i][j][:x][3])
    end
end

df_alternativas = DataFrame()
df_alternativas.nomes = repeat(1:n_sims, inner=nM)
df_alternativas.ano = repeat(repeat(1:nA, inner=12), outer=n_sims)
df_alternativas.mes = repeat(repeat(1:12, outer=nA), outer=n_sims)
df_alternativas.retirada_reuso = retirada_reuso
df_alternativas.retirada_dessal = retirada_dessal
df_alternativas.retirada_pisf = retirada_pisf

CSV.write("results/df_alternativas.csv", df_alternativas)

# Calcular risco de falha

falha_hum_res = Array{Any}(undef, n_res)
falha_irr_res = Array{Any}(undef, n_res)
falha_hum_res_abs = Array{Any}(undef, n_res)
falha_irr_res_abs = Array{Any}(undef, n_res)

for k in 1:9
    falha_hum_res[k] = zeros(0)
    falha_irr_res[k] = zeros(0)    
    falha_hum_res_abs[k] = zeros(0)
    falha_irr_res_abs[k] = zeros(0)
end

for i in 1:n_sims
    for j in 1:nM
        append!(falha_hum_res[1], 100*(demand_hum[1][j] - sims[i][j][:x_hum][1])/demand_hum[1][j])
        append!(falha_hum_res[2], 0)
        append!(falha_hum_res[3], 100*(demand_hum[3][j] - (sims[i][j][:x][1] + sims[i][j][:x_hum][3] + sims[i][j][:x_gav_pecem]))/demand_hum[3][j])
        append!(falha_hum_res[4], 100*(demand_hum[4][j] - sims[i][j][:x_hum][4])/demand_hum[4][j])
        append!(falha_hum_res[5], 100*(demand_hum[5][j] - sims[i][j][:x_hum][5])/demand_hum[5][j])
        append!(falha_hum_res[6], 0)
        append!(falha_hum_res[7], 0)
        append!(falha_hum_res[8], 0)
        append!(falha_hum_res[9], 100*(demand_hum[9][j] - (sims[i][j][:x][2] + sims[i][j][:x_hum][9]))/demand_hum[9][j])

        append!(falha_hum_res_abs[1], (demand_hum[1][j] - sims[i][j][:x_hum][1]))
        append!(falha_hum_res_abs[2], 0)
        append!(falha_hum_res_abs[3], (demand_hum[3][j] - (sims[i][j][:x][1] + sims[i][j][:x_hum][3] + sims[i][j][:x_gav_pecem])))
        append!(falha_hum_res_abs[4], (demand_hum[4][j] - sims[i][j][:x_hum][4]))
        append!(falha_hum_res_abs[5], (demand_hum[5][j] - sims[i][j][:x_hum][5]))
        append!(falha_hum_res_abs[6], 0)
        append!(falha_hum_res_abs[7], 0)
        append!(falha_hum_res_abs[8], 0)
        append!(falha_hum_res_abs[9], (demand_hum[9][j] - (sims[i][j][:x][2] + sims[i][j][:x_hum][9])))

        append!(falha_irr_res[1], 100*(demand_irr[1][j] - sims[i][j][:x_irr][1])/demand_irr[1][j])
        append!(falha_irr_res[2], 0)
        append!(falha_irr_res[3], 0)
        append!(falha_irr_res[4], 100*(demand_irr[4][j] - sims[i][j][:x_irr][4])/demand_irr[4][j])
        append!(falha_irr_res[5], 100*(demand_irr[5][j] - sims[i][j][:x_irr][5])/demand_irr[5][j])
        append!(falha_irr_res[6], 0)
        append!(falha_irr_res[7], 0)
        append!(falha_irr_res[8], 0)
        append!(falha_irr_res[9], 0)

        append!(falha_irr_res_abs[1], (demand_irr[1][j] - sims[i][j][:x_irr][1]))
        append!(falha_irr_res_abs[2], 0)
        append!(falha_irr_res_abs[3], 0)
        append!(falha_irr_res_abs[4], (demand_irr[4][j] - sims[i][j][:x_irr][4]))
        append!(falha_irr_res_abs[5], (demand_irr[5][j] - sims[i][j][:x_irr][5]))
        append!(falha_irr_res_abs[6], 0)
        append!(falha_irr_res_abs[7], 0)
        append!(falha_irr_res_abs[8], 0)
        append!(falha_irr_res_abs[9], 0)

    end
end

df = DataFrame()
df.nomes = repeat(repeat(1:100, inner=360), outer=n_res)
df.reservatorio = repeat(reservatorios.reservatorio, inner=n_sims*nM)
df.ano = repeat(repeat(repeat(1:nA, inner=12), outer=n_sims), outer=n_res)
df.mes = repeat(repeat(1:12, outer=3000), outer=n_res)
df.falha_irr = collect(Iterators.flatten(falha_irr_res))
df.falha_hum = collect(Iterators.flatten(falha_hum_res))
df.falha_abs_irr = collect(Iterators.flatten(falha_irr_res_abs))
df.falha_abs_hum = collect(Iterators.flatten(falha_hum_res_abs))

CSV.write("results/df_falha.csv", df)