Distance:=proc(s,t)
ans:=sqrt(add((s[i] - t[i])^2, i = 1 .. nops(s)));
return ans;
end proc;
