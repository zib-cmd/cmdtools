function [chi,fret]=opt_soft(evs,Ndim,kdim)

% Parameter:
%    evs:        Eigenvektoren
%                evs[l+Ndim*j] l-te Komponete vom j-ten EV
% 	       Dimension Ndim*kdim
%    chi:        (Ausgabe) Soft-characteristische Funktionen
%                chi[l+Ndim*j] l-te Komponetne von j-ter Funktion
% 	       Dimension: Ndim*kdim
%    Ndim:       Anzahl der Boxen
%    kdim:       Anzahl der Cluster

global NORMA
global MAXITER

N=Ndim(1); k=kdim(1);
flag=1;

if (k > 1)
    index=zeros(1,k);
    A=zeros(k,k);
    
    EVS=zeros(N,k);
    for l=1:N
      for j=1:k
          EVS(l,j) = evs(l+N*(j-1));
      endl,j
    end

    % Ecken des Simplex aussuchen global 
    index=indexsearch(EVS, N, k);

    
    % Die Transformationsmatrix A nach "alter Methode" 
    % als Startschaetzung fuer eine lokale Optimierung berechnen */
    A=EVS(index,:);
    A=inv(A);
    NORMA=norm(A(2:k,2:k));
    %A=RotMat;   %nicht zulässig!
    
   
    minchi_start=min(min(EVS*A))

    
    if (flag > 0)   
      %Das Optimierungsproblem laesst sich auf (k-1)^2 reduzieren 
        alpha=zeros(1,(k-1)^2);
        for i=1:k-1
            for j=1:k-1
                alpha(j + (i-1)*(k-1)) = A(i+1,j+1);
            end
        end
        startval=-objective(alpha,EVS,N,k,A)    %Startwert für zulässiges A
        % Optimierung durchfuehren 
        options=optimset('maxiter',MAXITER);
        [alpha,fret,exitflag,output]=fminsearch(@(alpha) objective(alpha,EVS,N,k,A),alpha,options);
        endval=-fret

        %Nach der Optimierung: A vollstaendig berechnen */
        for i=1:k-1
            for j=1:k-1
                A(i+1,j+1)=alpha(j + (i-1)*(k-1));
            end
        end
        A=fillA(A, EVS, N, k );
    else
        fret = -1;
    end
    

    chi=EVS*A;

else % Spezialfall: k==1
    if (flag==1)
      fret = 1.0;
    else
      fret = -1.0;
    end
    chi=ones(N,1);
    
end

  
