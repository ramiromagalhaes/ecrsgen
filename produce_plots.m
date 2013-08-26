f = figure('InvertHardcopy','off','Color',[1 1 1]);
axes1 = axes('Parent',f,'FontSize',8);
set(f,'Units','pixels','Position',[0 0 700 700]);  %# Modify figure size

files = dir('./fei-a/*.txt');
for file = files'
    filepath = ['./fei-a/' file.name]; 
    a = textread(filepath);
    scatter(a(:,1),a(:,2),3,'Parent',axes1,'DisplayName','a(:,2) vs. a(:,1)','YDataSource','a(:,2)');
    axis([0 1 0 1]);
    set(f, 'PaperUnits', 'inches', 'PaperPosition', [0 0 700 700] / 200);
    print(f, '-dpng', '-r200', ['images/' file.name '.png']);
end

