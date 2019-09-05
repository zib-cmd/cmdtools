function chi=optimizeMetastab(X,n)
%    X:          schur vectors
%    n:          number of Clusters

global ITERMAX


ITERMAX=500;    %500

N=size(X,1);
opts.disp=0;
EVS=X(:,1:n);

evs=reshape(EVS,N*n,1);
[chi,val]=opt_soft(evs, N, n);
