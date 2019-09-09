function EVS=orthogon(EVS,pi,lambda,N,k)
% make eigenvectors pi-orthonormal and the first column constant


perron=1;

% bring eigenvectors to correct length
pi=pi/sum(pi);
for i=1:k
    EVS(:,k)=EVS(:,k)/(sqrt((EVS(:,k).*pi)'*EVS(:,k))*sign(EVS(1,k)));
end

% count possible degenerations
for i=1:k
    if lambda(i) > 0.9999
      perron=perron+1;
    else
      break;
    end
end

if (perron > 1) % search for constant eigenvector
    maxscal=0.0;
    for i=1:perron
        scal=sum(pi.*EVS(:,i));
        if (abs(scal) > maxscal)
            maxscal = abs(scal);
            maxi=i;
        end
    end
    % shift non-constant eigenvectors to the last columns
    EVS(:,maxi)=EVS(:,1);
    EVS(:,1)=ones(N,1);
    EVS(find(pi<=0),:)=0;

    % pi-orthogonalize the other eigenvectors */
    for i=2:k
        for j=1:i-1
            scal = (EVS(:,j).*pi)'*EVS(:,i);
            EVS(:,i) = EVS(:,i)- scal * EVS(:,j);
        end
        sumval =sqrt((EVS(:,i).*pi)'*EVS(:,i));
        EVS(:,i)=EVS(:,i)/sumval;
    end
    
end

