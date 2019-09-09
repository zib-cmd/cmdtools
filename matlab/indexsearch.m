function index=indexsearch(evs,N,k)
% Zur Erzeugung einer Startloesung wird in den Daten eine
% moegliche Simplexstruktur gefunden.

maxdist=0.0;

rthoSys=zeros(N,N);
temp=zeros(1,N);

OrthoSys=evs;

% erste Ecke des Simplex: Normgroesster Eintrag */
for l=1:N
    dist = norm(OrthoSys(l,:));
    if (dist > maxdist)
        maxdist=dist;
	    index(1)=l;
    end
end

% for l=1:N
%     OrthoSys(l,:)=OrthoSys(l,:)-evs(index(1),:);
% end
OrthoSys=OrthoSys-ones(N,1)*evs(index(1),:);

% Alle weiteren Ecken jeweils mit maximalen Abstand zum
% bereits gewaehlten Unterraum */
for j=2:k
    maxdist=0.0;
    temp=OrthoSys(index(j-1),:);
    for l=1:N
        sclprod=OrthoSys(l,:)*temp';
        OrthoSys(l,:)=OrthoSys(l,:)-sclprod*temp;
        distt=norm(OrthoSys(l,:));
	    if (distt > maxdist ) %&& ~ismember(l,index(1:j-1))
            maxdist=distt;
            index(j)=l;
        end
    end
    OrthoSys = OrthoSys/maxdist;
end


   