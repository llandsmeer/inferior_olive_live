import matplotlib.pyplot as plt
import numba
import numpy as np

@numba.jit(fastmath=False, cache=True, nopython=True)
def simulate(skip_initial_transient_seconds=0, sim_seconds=10, delta=0.025, record_every=20,
        # Parameters
        g_int           =   0.13,    # Cell internal conductance  -- now a parameter (0.13)
        p1              =   0.25,    # Cell surface ratio soma/dendrite
        p2              =   0.15,    # Cell surface ratio axon(hillock)/soma
        g_CaL           =   1.1,     # Calcium T - (CaV 3.1) (0.7)
        g_h             =   0.12,    # H current (HCN) (0.4996)
        g_K_Ca          =  35.0,     # Potassium  (KCa v1.1 - BK) (35)
        g_ld            =   0.01532, # Leak dendrite (0.016)
        g_la            =   0.016,   # Leak axon (0.016)
        g_ls            =   0.016,   # Leak soma (0.016)
        S               =   1.0,     # 1/C_m, cm^2/uF
        g_Na_s          = 150.0,     # Sodium  - (Na v1.6 )
        g_Kdr_s         =   9.0,     # Potassium - (K v4.3)
        g_K_s           =   5.0,     # Potassium - (K v3.4)
        g_CaH           =   4.5,     # High-threshold calcium -- Ca V2.1
        g_Na_a          = 240.0,     # Sodium
        g_K_a           = 240.0,     # Potassium (20)
        V_Na            =  55.0,     # Sodium
        V_K             = -75.0,     # Potassium
        V_Ca            = 120.0,     # Low-threshold calcium channel
        V_h             = -43.0,     # H current
        V_l             =  10.0,     # Leak
        I_app           =   0.0,
        I_pulse10ms     =   0.0,
        I_noise_amp     =   0.0
        ):

    # Soma state
    V_soma          = -60.0
    soma_k          =   0.7423159
    soma_l          =   0.0321349
    soma_h          =   0.3596066
    soma_n          =   0.2369847
    soma_x          =   0.1

    # Axon state
    V_axon          = -60.0
    axon_Sodium_h   =   0.9
    axon_Potassium_x=   0.2369847

    # Dend state
    V_dend          = -60.0
    dend_Ca2Plus    =   3.715
    dend_Calcium_r  =   0.0113
    dend_Potassium_s=   0.0049291
    dend_Hcurrent_q =   0.0337836

    def _update_soma(iv_trace, at):
        'Perform a single soma timestep update'
        nonlocal V_soma, soma_k, soma_l, soma_h, soma_n, soma_x

        # CURRENT: Soma leak current (ls)
        soma_I_leak        = g_ls * (V_soma - V_l)

        # CURRENT: Soma interaction current (ds, as)
        I_ds        =  (g_int / p1)        * (V_soma - V_dend)
        I_as        =  (g_int / (1 - p2))  * (V_soma - V_axon)
        soma_I_interact =  I_ds + I_as

        # CHANNEL: Soma Low-threshold calcium (CaL)
        soma_Ical   = g_CaL * soma_k * soma_k * soma_k * soma_l * (V_soma - V_Ca)

        soma_k_inf  = 1 / (1 + np.exp(-(V_soma + 61)/4.2))
        soma_l_inf  = 1 / (1 + np.exp( (V_soma + 85)/8.5))
        soma_tau_l  = (20 * np.exp((V_soma + 160)/30) / (1 + np.exp((V_soma + 84) / 7.3))) + 35

        soma_dk_dt  = soma_k_inf - soma_k
        soma_dl_dt  = (soma_l_inf - soma_l) / soma_tau_l
        soma_k      = delta * soma_dk_dt + soma_k
        soma_l      = delta * soma_dl_dt + soma_l

        # CHANNEL: Soma sodium (Na_s)
        # watch out direct gate: m = m_inf
        soma_m_inf  = 1 / (1 + np.exp(-(V_soma + 30)/5.5))
        soma_h_inf  = 1 / (1 + np.exp( (V_soma + 70)/5.8))
        soma_Ina    = g_Na_s * soma_m_inf**3 * soma_h * (V_soma - V_Na)
        soma_tau_h  = 3 * np.exp(-(V_soma + 40)/33)
        soma_dh_dt  = (soma_h_inf - soma_h) / soma_tau_h
        soma_h      = soma_h + delta * soma_dh_dt

        # CHANNEL: Soma potassium, slow component (Kdr)
        soma_Ikdr   = g_Kdr_s * soma_n**4 * (V_soma - V_K)
        soma_n_inf  = 1 / ( 1 + np.exp(-(V_soma +  3)/10))
        soma_tau_n  = 5 + (47 * np.exp( (V_soma + 50)/900))
        soma_dn_dt  = (soma_n_inf - soma_n) / soma_tau_n
        soma_n      = delta * soma_dn_dt + soma_n

        # CHANNEL: Soma potassium, fast component (K_s)
        soma_Ik      = g_K_s * soma_x**4 * (V_soma - V_K)
        soma_alpha_x = 0.13 * (V_soma + 25) / (1 - np.exp(-(V_soma + 25)/10))
        soma_beta_x  = 1.69 * np.exp(-(V_soma + 35)/80)
        soma_tau_x_inv=soma_alpha_x + soma_beta_x
        soma_x_inf   = soma_alpha_x / soma_tau_x_inv

        soma_dx_dt   = (soma_x_inf - soma_x) * soma_tau_x_inv
        soma_x       = delta * soma_dx_dt + soma_x

        # RECORD: Soma variables
        if at >= 0:
            iv_trace[at, 0] = soma_Ik
            iv_trace[at, 1] = soma_Ikdr
            iv_trace[at, 2] = soma_Ina
            iv_trace[at, 3] = soma_Ical
            iv_trace[at, 4] = V_soma

        # UPDATE: Soma compartment update (V_soma)
        soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
        soma_dv_dt = S * (-(soma_I_leak + soma_I_interact + soma_I_Channels))
        V_soma = V_soma + soma_dv_dt * delta

    def _update_axon(iv_trace, at):
        'Perform a single axon hillock timestep update'
        nonlocal V_axon, axon_Sodium_h, axon_Potassium_x

        # CURRENT: Axon leak current (la)
        axon_I_leak    =  g_la * (V_axon - V_l)

        # CURRENT: Axon interaction current (sa)
        I_sa           =  (g_int / p2) * (V_axon - V_soma)
        axon_I_interact=  I_sa

        # CHANNEL: Axon sodium (Na_a)
        # watch out direct gate: m = m_inf
        axon_m_inf     =  1 / (1 + np.exp(-(V_axon+30)/5.5))
        axon_h_inf     =  1 / (1 + np.exp( (V_axon+60)/5.8))
        axon_Ina       =  g_Na_a * axon_m_inf**3 * axon_Sodium_h * (V_axon - V_Na)
        axon_tau_h     =  1.5 * np.exp(-(V_axon+40)/33)
        axon_dh_dt     =  (axon_h_inf - axon_Sodium_h) / axon_tau_h
        axon_Sodium_h  =  axon_Sodium_h + delta * axon_dh_dt

        # CHANNEL: Axon potassium (K_a)
        axon_Ik        =  g_K_a * axon_Potassium_x**4 * (V_axon - V_K)
        axon_alpha_x   =  0.13*(V_axon + 25) / (1 - np.exp(-(V_axon + 25)/10))
        axon_beta_x    =  1.69 * np.exp(-(V_axon + 35)/80)
        axon_tau_x_inv =  axon_alpha_x + axon_beta_x
        axon_x_inf     =  axon_alpha_x / axon_tau_x_inv
        axon_dx_dt     =  (axon_x_inf - axon_Potassium_x) * axon_tau_x_inv
        axon_Potassium_x= delta * axon_dx_dt + axon_Potassium_x

        # RECORD: Axon variables
        if at >= 0:
            iv_trace[at, 5] = axon_Ina
            iv_trace[at, 6] = axon_Ik
            iv_trace[at, 7] = V_axon

        # UPDATE: Axon hillock compartment update (V_axon)
        axon_I_Channels = axon_Ina + axon_Ik
        dv_dt  = S * (-(axon_I_leak +  axon_I_interact + axon_I_Channels))
        V_axon = V_axon + dv_dt * delta

    def _update_dend(iv_trace, at, t):
        'Perform a single denrite timestep update'
        nonlocal V_dend, dend_Ca2Plus, dend_Calcium_r, dend_Potassium_s, dend_Hcurrent_q

        # CURRENT: Dend application current (I_app, I_pulse10ms)
        dend_I_application = -I_app + (-I_pulse10ms if \
                 200 * sim_seconds < t - 1000 * skip_initial_transient_seconds < 210 * sim_seconds \
                else 0) + I_noise_amp * 5 * np.random.normal(0, 1)

        # CURRENT: Dend leak current (ld)
        dend_I_leak     =  g_ld * (V_dend - V_l)

        # CURRENT: Dend interaction Current (sd)
        dend_I_interact =  (g_int / (1 - p1)) * (V_dend - V_soma)

        # CHANNEL: Dend high-threshold calcium (CaH)
        dend_Icah       =  g_CaH * dend_Calcium_r * dend_Calcium_r * (V_dend - V_Ca)
        dend_alpha_r    =  1.7 / (1 + np.exp(-(V_dend - 5)/13.9))
        dend_beta_r     =  0.02*(V_dend + 8.5) / (np.exp((V_dend + 8.5)/5) - 1.0)
        dend_tau_r_inv5 =  (dend_alpha_r + dend_beta_r) # tau = 5 / (alpha + beta)
        dend_r_inf      =  dend_alpha_r / dend_tau_r_inv5
        dend_dr_dt      =  (dend_r_inf - dend_Calcium_r) * dend_tau_r_inv5 * 0.2
        dend_Calcium_r  =  delta * dend_dr_dt + dend_Calcium_r

        # CHANNEL: Dend calcium dependent potassium (KCa)
        dend_Ikca       =  g_K_Ca * dend_Potassium_s * (V_dend - V_K)
        dend_alpha_s    =  (0.00002 * dend_Ca2Plus) * (0.00002 * dend_Ca2Plus < 0.01) + 0.01*(0.00002 * dend_Ca2Plus > 0.01)
        dend_tau_s_inv  =  dend_alpha_s + 0.015
        dend_s_inf      =  dend_alpha_s / dend_tau_s_inv
        dend_ds_dt      =  (dend_s_inf - dend_Potassium_s) * dend_tau_s_inv
        dend_Potassium_s=  delta * dend_ds_dt + dend_Potassium_s

        # CHANNEL: Dend proton (h)
        dend_Ih         =  g_h * dend_Hcurrent_q * (V_dend - V_h)
        q_inf           =  1 / (1 + np.exp((V_dend + 80)/4))
        tau_q_inv       =  np.exp(-0.086*V_dend - 14.6) + np.exp(0.070*V_dend - 1.87)
        dq_dt           =  (q_inf - dend_Hcurrent_q) * tau_q_inv
        dend_Hcurrent_q =  delta * dq_dt + dend_Hcurrent_q

        # CONCENTRATION: Dend calcium concentration (CaPlus)
        dCa_dt          =  -3 * dend_Icah - 0.075 * dend_Ca2Plus
        dend_Ca2Plus    =  delta * dCa_dt + dend_Ca2Plus

        # RECORD: Dend variables
        if at >= 0:
            iv_trace[at, 8] = dend_Icah
            iv_trace[at, 9] = dend_Ikca
            iv_trace[at, 10] = dend_Ih
            iv_trace[at, 11] = V_dend

        # UPDATE: Dend compartment update (V_dend)
        dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
        dend_dv_dt  = S * (-(dend_I_leak +  dend_I_interact + dend_I_application + dend_I_Channels))
        V_dend = V_dend + dend_dv_dt * delta

    # Initialize simulation
    nepochs = int(sim_seconds*1000 / delta / record_every + .5)
    iv_trace = np.empty((nepochs, 13))
    nskip = int(1000 * skip_initial_transient_seconds / delta + 0.5)
    t = 0

    # Transient simulation loop
    for _i_skip in range(nskip):
        _update_soma(iv_trace, -1)
        _update_axon(iv_trace, -1)
        _update_dend(iv_trace, -1, t)
        t += delta

    # Recorded simulation loop
    for i_epoch in range(nepochs):
        for _i_ts in range(record_every):
            _update_soma(iv_trace, i_epoch)
            _update_axon(iv_trace, i_epoch)
            _update_dend(iv_trace, i_epoch, t)
            t += delta
        iv_trace[i_epoch, -1] = t

    # Return simulation results
    return iv_trace

def main():
    for I_app in np.linspace(0, 1, 3):
        iv_trace = simulate(skip_initial_transient_seconds=1, sim_seconds=1, I_app=I_app)
        (soma_Ik, soma_Ikdr, soma_Ina, soma_Ical, V_soma,
         axon_Ina, axon_Ik, V_axon,
         dend_Icah, dend_Ikca, dend_Ih, V_dend, t) = iv_trace.T
        plt.plot(t, V_dend, label=f'I_app={I_app}', alpha=I_app/2+0.5, color='k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Soma potential (mV)')
    plt.show()

if __name__ == '__main__':
    main()
