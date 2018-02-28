cs_amp test

***************************// Device Model //********************************
.model  nch nmos level=1    vto=0.7     gamma=0.45  phi=0.9
+nsub=9e14  ld=0.08e-6  uo=350  lambda=0.1  tox=9e-9    pb=0.9
+cj=0.56e-3 cjsw=0.35e-11   mj=0.45 mjsw=0.2    cgdo=0.4e-9 js=1.0e-8

.param  L=0.18u     E=0.54u     vbias=0.8   rload=100
.param  nf=10   W=0.5u  as="W*E"    ad="W*E"    ps="2*(E+W)"    pd="2*(E+W)"

M1  vd  vg  0   0   nch  w=W  l=L  m=10 as=as   ad=ad   ps=ps   PD=pd

Rl  VDD vd rload

Vdd VDD 0   1.8
vin vg  0   dc=vbias    ac=1

.op
.ac dec 20  1Meg  100G

.control 
*set filetype=ascii
save db(vd)
run 
*write output.txt i(Vdd)
.endc

.end
