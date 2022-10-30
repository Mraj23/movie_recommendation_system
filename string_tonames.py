def a(rows):
    x = rows["keywords"]
    stored_words = []
    semi_count = 0
    prev_semiindex = 0
    for i in range(len(x)):
        element = x[i]
        if element == ':':
            semi_count += 1
            if semi_count%2 == 0:
                prev_semiindex = i
        if element == "}":
            print(prev_semiindex,i)
            stored_words.append(x[prev_semiindex+3:i-1])
    return stored_words
  
  if __name__ == "__main__":
    main()
  
  
