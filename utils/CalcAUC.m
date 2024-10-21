function [ auc ] = CalcAUC( train, test, sim, n )    %��û�зǽ����㷨��
%% ����AUC�������������ƶȾ���
    sim = triu(sim - sim.*train) - diag(diag(sim));
    % ֻ�������Լ��Ͳ����ڱ߼����еıߵ����ƶȣ��Ի����⣩
    non = 1 - train - test - eye(max(size(train,1),size(train,2)));
    test = triu(test);
    non = triu(non);
    % �ֱ�ȡ���Լ��Ͳ����ڱ߼��ϵ������Ǿ�������ȡ�����Ƕ�Ӧ�����ƶȷ�ֵ
    test_num = nnz(test);
    non_num = nnz(non);
%     test_rd = ceil( test_num * rand( 1, n));  
%     % ceil��ȡ���ڵ��ڵ���С������nΪ�����ȽϵĴ���
%     non_rd = ceil( non_num * rand( 1, n));
    test_pre = sim .* test;
    non_pre = sim .* non;
    test_data =  test_pre( test ~= 0 )';  
    % ��������test ���ϴ��ڵıߵ�Ԥ��ֵ
    non_data =  non_pre( non ~= 0 )';   
    % ��������nonexist���ϴ��ڵıߵ�Ԥ��ֵ
%     test_rd = test_data( test_rd );
%     non_rd = non_data( non_rd );
%     %clear test_data non_data;
%     n1 = length( find(test_rd > non_rd) );  
%     n2 = length( find(test_rd == non_rd));
%     auc = ( n1 + 0.5*n2 ) / n;
    
    % matlab�Դ�����perfcurve
    labels = [ones(1,size(test_data,2)), zeros(1,size(non_data,2))];
    scores = [test_data, non_data];
    [X,Y,T,auc] = perfcurve(labels, scores, 1);
    
    
    
    
    %MATLAB calculate confusion matrix, ����ʵ��ʱע�͵�
    for runthis = 1:0
    ratio = 1;
    labels = [ones(1,size(test_data,2)), 2*ones(1,size(non_data,2))];
    scores = [test_data, non_data];
    [y,i] = sort(scores,2,'descend');
    y = y(:,1:test_num*ratio);
    i = i(:,1:test_num*ratio);
    g1 = labels(i);
    g2 = ones(test_num*ratio,1);
    C = confusionmat(g1,g2)
    precision = C(1,1)/test_num
    
    figure(1);
    plot(X,Y)
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('ROC');
    end
end
