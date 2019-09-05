function optval=objective(alpha,X,N,k,A)

% Hier wird die Zielfunktion berechnet, die die Spur der
% Massenmatrix maximiert. 

global NORMA  
global OPT




%Bestimmung der vollstaendigen Matrix A
for i=1:k-1
    for j=1:k-1
      A(i+1,j+1)=alpha(j + (i-1)*(k-1));
    end
end

normA=norm(A(2:k,2:k));

%A zulässig machen
A=fillA(A, X, N, k );
nc=size(X,2);

    J2=trace(diag(1./A(1,:))*A'*A);  %traceS
    optval=J2;

optval=-optval;
  
