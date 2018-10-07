% export imdb.mat to csv file

load imdb.mat

csvfile = fopen("imdb.csv", "w");

fprintf(csvfile, "dob,photo_taken,full_path,gender,name,face_location,face_score,second_face_score,celeb_id\n");

for i = 1:length(imdb.dob)
   if (i == 10000)
       break
   end
   
    fprintf(csvfile, "%d,", imdb.dob(i));
    fprintf(csvfile, "%d,", imdb.photo_taken(i));
    
    celldata = cellstr(imdb.full_path(i));
    strdata = char(celldata);
    fprintf(csvfile, "%s,", strdata);
    
    clear strdata;
    
    fprintf(csvfile, "%d,", imdb.gender(i));
    
    celldata = cellstr(imdb.name(i));
    strdata = char(celldata);
    fprintf(csvfile, "%s,", strdata);
    clear strdata;
    
    celldata = cell2mat(imdb.face_location(i));
    strdata = mat2str(celldata);
    fprintf(csvfile, "%s,", strdata);
    clear strdata;
    
    fprintf(csvfile, "%f,", imdb.face_score(i));
    fprintf(csvfile, "%f,", imdb.second_face_score(i));
    fprintf(csvfile, "%d \n", imdb.celeb_id(i));
    
     disp(i);
end

fclose(csvfile);