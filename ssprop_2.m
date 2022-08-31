function [u1, u2] = ssprop_2(u0,u00,dt,dz,nz,alpha,betap,gamma)

% This function solves the nonlinear Schrodinger equation for
% pulse propagation in an optical fiber using the split-step
% Fourier method.
% 
% The following effects are included in the model: group velocity
% dispersion (GVD), higher order dispersion, loss, and self-phase
% modulation (gamma).
% 
% USAGE
%
% u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma);
% u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma,maxiter);
% u1 = ssprop(u0,dt,dz,nz,alpha,betap,gamma,maxiter,tol);
%
% INPUT
%
% u0 - starting field amplitude (vector)
% dt - time step
% dz - propagation stepsize
% nz - number of steps to take, ie, ztotal = dz*nz
% alpha - power loss coefficient, ie, P=P0*exp(-alpha*z)
% betap - dispersion polynomial coefs, [beta_0 ... beta_m]
% gamma - nonlinearity coefficient
% maxiter - max number of iterations (default = 4)
% tol - convergence tolerance (default = 1e-5)
%
% OUTPUT
%
% u1 - field at the output
% 
% NOTES  The dimensions of the input and output quantities can
% be anything, as long as they are self consistent.  E.g., if
% |u|^2 has dimensions of Watts and dz has dimensions of
% meters, then gamma should be specified in W^-1*m^-1.
% Similarly, if dt is given in picoseconds, and dz is given in
% meters, then beta(n) should have dimensions of ps^(n-1)/m.
%
% See also:  sspropc (compiled MEX routine)
%
% AUTHOR:  Thomas E. Murphy (tem@umd.edu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright 2006, Thomas E. Murphy
%
%   This file is part of SSPROP.
%
%   SSPROP is free software; you can redistribute it and/or
%   modify it under the terms of the GNU General Public License
%   as published by the Free Software Foundation; either version
%   2 of the License, or (at your option) any later version.
%
%   SSPROP is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public
%   License along with SSPROP; if not, write to the Free Software
%   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
%   02111-1307 USA

if (nargin<9)
  tol = 1e-5;
end
if (nargin<8)
  maxiter = 4;
end
flag=0;
nt = (length(u0));
if (mod(nt,2)~=0)
    u0=[u0;0];
    u00=[u00;0];
    flag=1;
    nt = length(u0);
end

% ch=size(u0)
% lpc=(-L/2:1:L/2-1)';
%w = double(2*pi*[(0:nt/2-1),(-nt/2:-1)]'/(dt*nt));
w = double(2*pi*[(0:nt/2-1),(-nt/2:-1)])'/double(dt*nt);
% w = 2*pi*[(-nt/2:1:nt/2-1)]'/(dt*nt);

% size(w)
halfstep = exp((double(-alpha/2) + 0.5*1i*betap(3)*(w).^2)*double(dz/2));

u1 = u0;
u2 = u00;
ufft = fft(u0);
ufft0 = fft(u00);
for iz = 1:nz
  uhalf = ifft(halfstep.*ufft);  %Xpol
  uhalf0 = ifft(halfstep.*ufft0); %Ypol
  uv = uhalf .*exp(1i*(8*(gamma/9))*(abs(uhalf).^2 +abs(uhalf0).^2 ) *double(dz));
  uv0 = uhalf0 .* exp(1i*(8*gamma/9)*(abs(uhalf).^2 +abs(uhalf0).^2 ) *double(dz));
  uv = fft(uv);
  uv0 = fft(uv0);
  ufft = halfstep.*uv;
  ufft0 = halfstep.*uv0;
  uv = ifft(ufft);
  uv0 = ifft(ufft0);
  u1 = uv;
  u2 = uv0;
end
if flag==1
    u1=u1(1:end-1);
    u2=u2(1:end-1);
end
% Ein=transpose(Ein);
% X_in=(u0);
% Y_in=(u00);
% Nc = length(X_in);
% Delta_omega=1/Nc/delta_t*2*pi;
% w = (-Nc/2:1:Nc/2-1)*Delta_omega;
% 
% % DBP
% X_buf = X_in;
% Y_buf = Y_in;
% lin_inv_halfstep = exp( (-alpha/2 - (1i*b_2/2).*w.^2) * (dz/2));
% lin_inv_halfstep=fftshift(lin_inv_halfstep);
% 
%     for i_st = 1:nz
%         X_buf = ifft(fft(X_buf) .* lin_inv_halfstep);
%         Y_buf = ifft(fft(Y_buf) .* lin_inv_halfstep);
%         nl_inv_step = exp((-1i*8/9*gamma*dz) * (abs(X_buf).^2 + abs(Y_buf).^2));
%         X_buf = X_buf .* nl_inv_step;
%         Y_buf = Y_buf .* nl_inv_step;
%         X_buf = ifft(fft(X_buf) .* lin_inv_halfstep);
%         Y_buf = ifft(fft(Y_buf) .* lin_inv_halfstep);
%     end
% 
% 
% % Ein=transpose(Ein);
% % Ein2=transpose(Ein2);
% % 
% % Delta_omega=1/Nc/delta_t*2*pi;
% % omega = (-Nc/2:1:Nc/2-1)*Delta_omega;
% % H=exp(1i*(b_2/2)*N_sp*L_sp*(omega.^2));
% % H=fftshift(H);
% % Eout=ifft(fft(X_buf).*H);
% % Eout2=ifft(fft(Y_buf).*H);
% % Eout=ifft(fft(Ein).*H);
% % Eout2=ifft(fft(Ein2).*H);
% % 
% % X_buf = Eout;
% % Y_buf = Eout2;
% 
% u1=(X_buf);
% u2=(Y_buf);

end